# /Users/murseltasgin/projects/chat_rag/components/chunker/semantic_chunker.py
"""
Semantic chunker with context preservation
"""
from typing import List, Dict, Any, Tuple
import numpy as np
from datetime import datetime
import nltk
from nltk.tokenize import sent_tokenize, TextTilingTokenizer
from .base import BaseChunker
from core.models import DocumentChunk
from core.exceptions import ChunkerException

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


class SemanticChunker(BaseChunker):
    """Semantic chunker with context preservation"""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        min_chunk_size: int = 100,
        use_semantic_segmentation: bool = True,
        use_embedding_segmentation: bool = False,
        semantic_threshold: float = 0.6,
        semantic_window: int = 1,
        max_tokens: int = None,
        token_overlap: int = None,
        next_context_chars: int = 140

    ):
        """
        Initialize semantic chunker
        
        Args:
            chunk_size: Maximum words per chunk
            chunk_overlap: Overlapping words between chunks
            min_chunk_size: Minimum words per chunk
            use_semantic_segmentation: If True, split by topic using TextTiling before sentence packing
            use_embedding_segmentation: If True, detect boundaries via sentence embedding similarity
            semantic_threshold: Cosine similarity threshold to cut segments (< threshold triggers split)
            semantic_window: Number of sentences to average on each side for stability
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.use_semantic_segmentation = use_semantic_segmentation
        self.use_embedding_segmentation = use_embedding_segmentation
        self.semantic_threshold = semantic_threshold
        self.semantic_window = semantic_window

        self.max_tokens = max_tokens
        self.token_overlap = (
            token_overlap if token_overlap is not None
            else (max_tokens // 5 if max_tokens else None)
        )
        self.next_context_chars = next_context_chars
        self._tokenizer = None
    
    def chunk_text(
        self,
        text: str,
        doc_id: str,
        doc_title: str,
        document_summary: str = None,
        **kwargs
    ) -> List[DocumentChunk]:
        """Chunk text into document chunks"""
        try:
            from components.document_processor import DocumentProcessor
            
            processor = DocumentProcessor()
            
            if self.max_tokens and self._tokenizer is None:
                embedding_model = kwargs.get('embedding_model')
                if embedding_model is not None:
                    self._tokenizer = getattr(
                        getattr(embedding_model, "model", None), "tokenizer", None
                    )
                if self._tokenizer is None:
                    print("No tokenizer available; falling back to word based chunk sizing")

            sections = processor.extract_sections(text)

            all_chunks = []
            chunk_counter = 0
            
            for section_title, section_content in sections:
                cleaned = processor.clean_text(section_content)
                # First, apply semantic segmentation into topical segments (if enabled)
                if self.use_embedding_segmentation and kwargs.get('embedding_model') is not None:
                    segments = self._semantic_segment_by_embeddings(
                        cleaned,
                        kwargs.get('embedding_model')
                    )
                else:
                    segments = self._semantic_segment(cleaned) if self.use_semantic_segmentation else [cleaned]

                section_chunks = []
                for seg in segments:
                    section_chunks.extend(self._chunk_by_sentences(seg, section_title))
                
                for chunk_data in section_chunks:
                    all_chunks.append({
                        'content': chunk_data['content'],
                        'section': chunk_data['section'],
                        'index': chunk_counter
                    })
                    chunk_counter += 1
            
            # If no chunks were created (document too small), create one chunk with all text
            if not all_chunks:
                cleaned_text = processor.clean_text(text)
                if cleaned_text.strip():
                    all_chunks.append({
                        'content': cleaned_text,
                        'section': 'Main Content',
                        'index': 0
                    })
            
            # Create DocumentChunk objects with context
            document_chunks = []
            total_chunks = len(all_chunks)
            
            for i, chunk_data in enumerate(all_chunks):
                # Get surrounding context
                #prev_context = all_chunks[i-1]['content'][:200] if i > 0 else None
                #next_context = all_chunks[i+1]['content'][:200] if i < total_chunks - 1 else None
                prev_context = None
                next_context = (
                    all_chunks[i+1]['content'][:self.next_context_chars]
                    if i < total_chunks - 1 else None
                )

                # Build header and prepend to content to improve retrieval
                section = chunk_data['section'] or 'Main Content'
                header_lines = [
                    f"Title: {doc_title}",
                    f"Document: {doc_id}",
                    f"Section: {section}"
                ]
                header = "\n".join(header_lines) + "\n\n"
                content_with_header = header + chunk_data['content']
                
                doc_chunk = DocumentChunk(
                    chunk_id=f"{doc_id}_chunk_{i}",
                    content=content_with_header,
                    doc_id=doc_id,
                    doc_title=doc_title,
                    chunk_index=i,
                    total_chunks=total_chunks,
                    section_title=chunk_data['section'],
                    previous_context=prev_context,
                    next_context=next_context,
                    document_summary=document_summary,
                    metadata={
                        'created_at': datetime.now().isoformat(),
                        'word_count': len(content_with_header.split())
                    }
                )
                document_chunks.append(doc_chunk)
            
            return document_chunks
        except Exception as e:
            raise ChunkerException(f"Chunking failed: {e}")
    
    def _chunk_by_sentences(
        self,
        text: str,
        section_title: str = None
    ) -> List[Dict[str, Any]]:
        """Chunk text by sentences while respecting semantic boundaries"""
        sentences = sent_tokenize(text)
        chunks = []
        #current_chunk = []
        #current_length = 0
        use_tokens = self.max_tokens is not None and self._tokenizer is not None
        limit = self.max_tokens if use_tokens else self.chunk_size
        overlap_limit = self.token_overlap if use_tokens else self.chunk_overlap

        current, current_len = [], 0
        
        for sentence in sentences:
            s_len = self._count(sentence)

            # one sentence longer than the whole budget
            if s_len > limit:
                if current:
                    chunks.append({'content': ' '.join(current), 'section': section_title})
                    current, current_len = [], 0
                for piece in self._split_oversized(sentence, limit):
                    chunks.append({'content': piece, 'section': section_title})
                continue

            if current_len + s_len > limit and current:
                chunks.append({'content': ' '.join(current), 'section': section_title})

                # overlap: walk back from the end, keep whole sentences
                overlap, o_len = [], 0
                for s in reversed(current):
                    l = self._count(s)
                    if o_len + l <= overlap_limit:
                        overlap.insert(0, s)
                        o_len += l
                    else:
                        break
                current, current_len = overlap, o_len

            current.append(sentence)
            current_len += s_len

        # final leftover: keep it, or merge it into the previous chunk.
        # Never discard content.
        if current:
            tail = ' '.join(current)
            if self._count(tail) >= self.min_chunk_size or not chunks:
                chunks.append({'content': tail, 'section': section_title})
            else:
                chunks[-1]['content'] = chunks[-1]['content'] + ' ' + tail

        return chunks

    def _semantic_segment(self, text: str) -> List[str]:
        """Segment text into topical segments using TextTiling; fall back to whole text on failure."""
        # TextTiling works best on longer texts; guard for very short inputs
        if not text or len(text.split()) < max(self.min_chunk_size, 80):
            return [text]
        try:
            tokenizer = TextTilingTokenizer()
            segments = tokenizer.tokenize(text)
            # Post-process: strip and drop empty/very small segments
            processed = []
            for seg in segments:
                seg_clean = seg.strip()
                if len(seg_clean.split()) >= max(20, self.min_chunk_size // 2):
                    processed.append(seg_clean)
            return processed if processed else [text]
        except Exception:
            return [text]

    def _semantic_segment_by_embeddings(self, text: str, embedding_model: Any) -> List[str]:
        """
        Segment text by comparing consecutive sentence embeddings. We cut where similarity
        dips below a threshold. No sentences are split.
        """
        sentences = sent_tokenize(text)
        if len(sentences) < 3:
            return [text]
        try:
            # Compute sentence embeddings (L2-normalized)
            embeddings = embedding_model.encode(sentences, convert_to_tensor=False)
            if isinstance(embeddings, list):
                embeddings = np.array(embeddings)
            # Normalize
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
            embeddings = embeddings / norms

            # Compute rolling similarities between windows of sentences
            window = max(int(self.semantic_window), 1)
            left_avgs = []
            right_avgs = []
            for i in range(len(sentences)):
                l_start = max(0, i - window)
                l_end = i
                r_start = i
                r_end = min(len(sentences), i + window)
                if l_end > l_start:
                    left_avgs.append(embeddings[l_start:l_end].mean(axis=0))
                else:
                    left_avgs.append(embeddings[i])
                if r_end > r_start:
                    right_avgs.append(embeddings[r_start:r_end].mean(axis=0))
                else:
                    right_avgs.append(embeddings[i])

            left_avgs = np.vstack(left_avgs)
            right_avgs = np.vstack(right_avgs)
            # Cosine similarity since vectors are normalized
            sims = (left_avgs * right_avgs).sum(axis=1)

            # Determine cut points where similarity is low and we have enough words in current segment
            cut_indices = []
            current_words = 0
            for idx in range(1, len(sentences) - 1):
                current_words += len(sentences[idx - 1].split())
                # Enforce minimum segment size to avoid over-segmentation
                if current_words < max(self.min_chunk_size, 40):
                    continue
                if sims[idx] < float(self.semantic_threshold):
                    cut_indices.append(idx)
                    current_words = 0

            # Build segments from cut points
            if not cut_indices:
                return [text]
            segments = []
            start = 0
            for cut in cut_indices:
                seg_sentences = sentences[start:cut]
                segments.append(' '.join(seg_sentences))
                start = cut
            if start < len(sentences):
                segments.append(' '.join(sentences[start:]))

            # Filter tiny segments and merge if necessary
            filtered = []
            for seg in segments:
                if len(seg.split()) >= max(20, self.min_chunk_size // 2):
                    filtered.append(seg)
                elif filtered:
                    filtered[-1] = (filtered[-1] + ' ' + seg).strip()
                else:
                    filtered.append(seg)
            return filtered if filtered else [text]
        except Exception:
            return [text]
    
    def get_name(self) -> str:
        """Get the chunker name"""
        return "SemanticChunker"
    
    def get_config(self) -> dict:
        """Get the chunker configuration"""
        return {
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'min_chunk_size': self.min_chunk_size
        }

    def _count(self, text: str) -> int:
        """Length of text in the embedding model's tokens, or words as a fallback."""
        if self._tokenizer is not None:
            return len(self._tokenizer(text, add_special_tokens=False)["input_ids"])
        return len(text.split())

    def _split_oversized(self, sentence: str, limit: int) -> List[str]:
        """Split a single sentence that is longer than the whole budget.

        PDF tables and long lists arrive as one huge 'sentence' because there
        is no full stop, and those are exactly the places the hardest facts
        live. Without this they become one oversized chunk that the embedding
        model then truncates silently.
        """
        pieces, current = [], []
        for part in sentence.split(","):
            part = part.strip()
            if not part:
                continue
            candidate = ", ".join(current + [part])
            if current and self._count(candidate) > limit:
                pieces.append(", ".join(current))
                current = [part]
            else:
                current.append(part)
        if current:
            pieces.append(", ".join(current))

        out = []
        for p in pieces:
            if self._count(p) <= limit:
                out.append(p)
            else:
                words = p.split()
                step = max(1, int(len(words) * limit / max(1, self._count(p))))
                out.extend(" ".join(words[i:i + step]) for i in range(0, len(words), step))
        return out
