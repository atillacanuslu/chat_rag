# /Users/murseltasgin/projects/chat_rag/components/contextual_enhancer/contextual_enhancer.py
"""
Contextual enhancement for RAG
"""
from components.llm import BaseLLM
from core.models import DocumentChunk
from core.exceptions import LLMException
from utils.logger import RAGLogger, get_logger


class ContextualRAGEnhancer:
    """Enhances chunks with contextual information using LLM"""
    
    def __init__(self, llm_model: BaseLLM):
        """
        Initialize contextual enhancer
        
        Args:
            llm_model: LLM model instance
        """
        self.llm_model = llm_model
        self.logger = get_logger("ContextualRAGEnhancer")
    
    def generate_document_summary(self, document_text: str, doc_title: str) -> str:
        """
        Generate a concise summary of the entire document
        
        Args:
            document_text: Full document text
            doc_title: Document title
        
        Returns:
            Document summary
        """
        prompt = f"""Aşağıdaki belgeyi tek bir kısa cümleyle özetle (en fazla 20 kelime).
Belgenin dilinde yaz.
Belge başlığı: "{doc_title}"

{document_text[:3000]}

Özet:"""
        
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant that creates concise document summaries."},
                {"role": "user", "content": prompt}
            ]
            
            # Log LLM request/response
            RAGLogger.log_llm_request(self.logger, messages, 0.3, 200)
            response = self.llm_model.generate(messages, temperature=0.3, max_tokens=200)
            RAGLogger.log_llm_response(self.logger, response, success=True)
            return response
        except Exception as e:
            print(f"Error generating summary: {e}")
            return f"Document: {doc_title}"
    
        # token budget for each part of the embedding text
    SECTION_TOKENS = 20
    CONTENT_TOKENS = 370
    NEXT_TOKENS = 40
    SUMMARY_TOKENS = 50

    def set_tokenizer(self, tokenizer):
        """Give the enhancer the embedding model's tokenizer so the budgets
        below are enforced in the same unit the model truncates on."""
        self._tokenizer = tokenizer

    def _count(self, text: str) -> int:
        if getattr(self, "_tokenizer", None) is None:
            return len(text.split())
        return len(self._tokenizer(text, add_special_tokens=False)["input_ids"])

    def _clip(self, text: str, max_tokens: int) -> str:
        """Hard limit on a supporting part. Used for section title, next
        context and summary only, never for the chunk text."""
        if getattr(self, "_tokenizer", None) is None:
            return text
        ids = self._tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= max_tokens:
            return text
        return self._tokenizer.decode(ids[:max_tokens])

    def enrich_chunk_with_context(self, chunk: DocumentChunk) -> str:
        """Build the text that gets embedded.

        Content is never truncated here: the chunker sizes it, and an
        oversized chunk is a bug worth seeing rather than text worth losing.
        The supporting parts are clipped to their budgets.
        """
        parts = []

        # unique to this chunk, and short
        if chunk.section_title and chunk.section_title != "Main Content":
            parts.append(f"Section: {self._clip(chunk.section_title, self.SECTION_TOKENS)}")

        # chunk.content carries a Title/Document/Section header. Strip it here:
        # Title repeats on every chunk, Document is a random upload id, and
        # Section is already included above.
        body = chunk.content
        if body.startswith("Title:"):
            split = body.split("\n\n", 1)
            if len(split) == 2:
                body = split[1]

        # NOT clipped. If this fires, fix the chunker rather than the symptom.
        n = self._count(body)
        if n > self.CONTENT_TOKENS:
            self.logger.warning(
                "Chunk %s is %d tokens, over the %d token content budget; "
                "the embedding model will truncate it",
                chunk.chunk_id, n, self.CONTENT_TOKENS,
            )
        parts.append(f"Content: {body}")

        # genuinely follows this chunk; already trimmed by the chunker
        if chunk.next_context:
            parts.append(f"Following: {self._clip(chunk.next_context, self.NEXT_TOKENS)}")

        # identical on every chunk of the document. Its only job is to identify
        # the document, which matters once the store holds more than one.
        if chunk.document_summary:
            parts.append(f"Belge: {self._clip(chunk.document_summary, self.SUMMARY_TOKENS)}")

        return " | ".join(parts)

