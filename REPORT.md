KKB RAG Challenge — Report

Atilla Can Uslu · September 2026 Repository: fork of MurselTasgin/chat_rag, branch improvements

1. Summary
The starting repository answered roughly one question in five correctly and there was no evaulation metric available. 
Therefore, the first set of changes was the addition of a 32 question evaluation set built from the KKB 2024 annual report and a runner to test the validity RAG chatbot. The runner scored retrieval and answers separetely, and recorded per stage scores(for hybrid_search, vector_search, BM25_search, and the reranker), latency, and the token usage without changing any behavior.

As a result of the first run of the we were able to clearly see two major problems I had already identified when analysing the code base:
    * BM25 always returned top_k chunks even in the case where all the chunks scored 0 in the BM25(meaning no keyword match), which resulted in a "division by 0 error" resulting in the hybrid_search crashing when normalizing the scors.
    * On my run in the MACbook, the reranker always crashed silently result in Nan scores for all the reranked chunks. However, since there is a mechanism in the code to return all the chunks with a 0.5 relevance score.

The remaining changes targeted retrieval and chunking quality, one reliability bug, and token cost.


2 How the Measurement works
The RAG system can fail in two different places. Retrieval can hand the model the wrong chunks, or the model might receive the right chunks and still provide an "incorrect" response.

The measurement focuses on these two parts separately, on the same run, for the same set of questions. Everything else follows from that decision.

2.1 The evaluation set

eval/questions.jsonl, A set of 32 questions targeted towards the KKB 2024 Faaliyet Raporu(the file present in KB3).Each questions comes with the answer we expect to get and one or more evidence phrases taken from the parsed document, so that the retrieval can be scored independently of the answer.


The questions set includes 13 numeric, 11 factual, 3 English(given that the document is in Turkish), 1 list, 1 multi hop, and 2 with no answer in the document. The question set was created with a mixture of different type of questions in order to test the system's ability to handle them.


2.2 Retrieval and the Response Check

Retrieval Check
In order to analyze if the right chunks actually reach the LLM, we conduct a basic string match of the evidence phrase against the final chunks returned from the reranker. It is a very simple, deterministic, repetable way of whether the LLM is actually getting the correct chunks. The evidence phrases were changed and made to accept variations of expected answer as the model often responded to the same questions in different ways.

Response Check


Strategy Call not doing anything. 
Incusion Rule.

