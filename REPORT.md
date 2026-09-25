KKB RAG Challenge — Report

Atilla Can Uslu · September 2026 Repository: fork of MurselTasgin/chat_rag, branch improvements

1. Summary
The starting repository answered roughly one question in five correctly and there was no evaulation metric available. 
Therefore, the first set of changes was the addition of a 32 question evaluation set built from the KKB 2024 annual report and a runner to test the validity RAG chatbot. The runner scored retrieval and answers separetely, and recorded per stage scores(for hybrid_search, vector_search, BM25_search, and the reranker), latency, and the token usage without changing any behavior.

As a result of the first run of the we were able to clearly see two major problems I had already identified when analysing the code base:
    * BM25 always returned top_k chunks even in the case where all the chunks scored 0 in the BM25(meaning no keyword match), which resulted in a "division by 0 error" resulting in the hybrid_search crashing when normalizing the scors.
    *On my run in the MACbook, the reranker always crashed silently


That measurement then found three failures that produce no error message and are invisible without it:

the cross encoder reranker returned NaN for every candidate on every query, so reranking was a no-op and every source displayed "Relevance 50.0%"

the chunker discarded any section shorter than its minimum, so two facts in the source document existed in zero of the 151 chunks and were unanswerable no matter how good retrieval became

BM25 returned chunks that shared no term with the query as keyword matches, which both polluted the candidate pool and crashed one question on every run

Headline result, same 32 questions throughout: