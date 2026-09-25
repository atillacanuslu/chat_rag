# KKB RAG Challenge — Report

Atilla Can Uslu · September 2026
Repository: fork of `MurselTasgin/chat_rag`, branch `improvements`

---

## 1. Summary

The starting repository answered roughly one question in five correctly and
there was no evaluation metric available.

Therefore, the first set of changes was the addition of a 32 question
evaluation set built from the KKB 2024 annual report and a runner to test the
validity of the RAG chatbot. The runner scored retrieval and answers
separately, and recorded per stage scores (for `hybrid_search`,
`vector_search`, BM25 search, and the reranker), latency, and token usage
without changing any behaviour.

The first run showed two major problems I had already suspected from reading
the code base:

- BM25 always returned `top_k` chunks even when every chunk scored 0, meaning
  no keyword match at all. That produced a division by zero when
  `hybrid_search` normalised the scores, and the query crashed.
- On the MacBook the reranker silently returned NaN for every chunk. A
  fallback in the code then gave all of them a flat 0.5 relevance score, so
  nothing failed visibly and the reranking stage was doing nothing.

The remaining changes targeted retrieval and chunking quality, one reliability
bug, and token cost.

---

## 2. How the Measurement works

The RAG system can fail in two different places. Retrieval can hand the model
the wrong chunks, or the model might receive the right chunks and still
provide an incorrect response.

The measurement scores these two parts separately, on the same run, for the
same set of questions. Everything else follows from that decision.

All results of a measurement run are written to a CSV and an Excel file at the
end of the run.

### 2.1 The evaluation set

`eval/questions.jsonl`, a set of 32 questions targeted at the KKB 2024
Faaliyet Raporu. Each question comes with the answer we expect to get and one
or more evidence phrases taken from the parsed document, so that retrieval can
be scored independently of the answer.

The question set includes 13 numeric, 12 factual, 3 English (given that the
document is in Turkish), 1 list, 1 multi hop, and 2 with no answer in the
document. It was built with a mixture of question types in order to test the
system's ability to handle each of them.

### 2.2 Retrieval and the Response Check

**Retrieval Check**

To analyse whether the right chunks actually reach the LLM, we do a plain
string match of the evidence phrases against the final chunks returned from
the reranker. It is simple, deterministic and repeatable, and it tells us
whether the LLM is getting the correct chunks. Each question accepts a list of
evidence phrases rather than one, because the document states the same fact in
more than one place and in more than one form.

**Response Check**

Since the expected answer is provided in the question set, I used an LLM judge
to compare it against the answer the model produced. The judge never sees the
chunks or the retrieval methods, it only compares the generated response with
the expected one. In the MacBook run with llama3.2 the judge was very
unreliable, while qwen2.5:7b on the PC was much better.

### 2.3 What is measured?

In addition to the retrieval check and the LLM judge, a number of metrics are
recorded at the end of each run:

| Metric | Definition |
|---|---|
| `hit_final` | 1 if an evidence phrase appears in the final `top_k` chunks, else 0 |
| Hit rate @5 | share of scored questions with `hit_final = 1`. Questions that crashed are excluded from the denominator |
| `rank` | position of the first chunk containing an evidence phrase, 1 to 5, or 0 if absent |
| MRR | mean of `1/rank` over scored questions, counting a miss as 0 |
| Answer accuracy | share of questions the judge marks correct |
| Crash rate | percentage of questions that raised an exception instead of returning an answer |
| Median and p90 latency | seconds per question, wall clock |
| Total run time | wall clock for the whole run |
| Input and output tokens | read from Ollama's `prompt_eval_count` and `eval_count`, reported as mean, median and total |
| LLM calls per question | 3 with conversation disabled: strategy, query generation, answer |
| Score spreads | max minus min per query, for `vector_search`, BM25 and `hybrid_search` |
| BM25 zero-score results (avg) | average number of returned keyword matches scoring exactly 0 |
| Reranker NaN count | number of NaN scores produced by the reranker |

### 2.4 Matching Turkish text

`eval/normalize.py` handles two things that silently break naive matching:

- `İ` maps to `i` and `I` to `ı` **before** lowercasing. Python's `lower()`
  turns `İ` into `i` plus a combining dot, so `"İstanbul".lower()` does not
  equal `"istanbul"`.
- Apostrophes are deleted rather than replaced with a space, because the
  pipeline's own `clean_text` deletes them: `KKB'nin` becomes `KKBnin`.

Without the second rule, every question mentioning `KKB'nin` or `Türkiye'nin`
scores as a miss even when retrieval worked perfectly.

### 2.5 The First Runs and What They Meant

Before building the Excel view, I ran a small three question set just to check
that `eval/run_eval.py` was doing what I thought it was doing. That run is what
justified the per stage score columns, because two things were visibly wrong
and neither of them would have shown up in a hit rate.

The minimum and the maximum reranker score were the same number on every
question. I added a NaN counter to the reranker stats and ran the three
questions again: 74 NaN scores across 3 questions, which is every score it
produced. The cross-encoder was returning NaN for every pair, the code fell
back to a constant 0.5, and the fallback fired on 31 of the 32 questions in the
full baseline run.

So the reranking stage, which is the last thing standing between retrieval and
the LLM, was doing nothing at all.

Results of this are in the file `cols.csv`.

---

## 3. Every reranker score was identical

As described in the previous section, the raw reranker scores read NaN on
every question and every final score was exactly 0.5, with the tie fallback
firing in all cases.

**Cause.** The reranker forced the model onto the CPU:

```python
device = device or 'cpu'
```

On Apple Silicon that returns NaN. Same model, same input:

| Where | Device | Score |
|---|---|---|
| MacBook Air M2 | cpu | nan |
| MacBook Air M2 | mps | 1.1546988 |
| Windows | cpu | 1.1546983 |

**Fix.**

```python
self.model = CrossEncoder(model_name, device=device)
try:
    self.model.model.float()
except Exception:
    pass
```

**Result.** After fixing the issue I ran the full eval over all 32 questions to
see the performance of the system. Although there was no change in hit rate or
answer accuracy, the share of queries with NaN reranker scores dropped to 0%,
which confirmed the fix worked.

I also added the full set of metrics described above to `eval/run_eval.py` in
the meantime, which is what let me identify the next set of changes.

The results of the run are available in `mac_baseline_v2.xlsx`, and a small
section of it is shown below:

| | Hit rate | MRR | Tie fallback | NaN |
|---|---|---|---|---|
| Before | 20.7% | 0.056 | 31 / 32 | every pair |
| After | 20.7% | 0.119 | 0 | 0 |

The hit rate does not move, and it should not. A reranker reorders the
candidates it is given, it does not retrieve new ones, so it cannot put a chunk
in the top five that fusion never passed to it. What it changes is where in the
five a chunk lands, which is what MRR measures.

---

## 4. The BM25 crash

After looking at `mac_baseline_v2.xlsx` I saw that the crash rate of the run
was 3.1%, with the following error:

```
q14  RAGException: Retrieval failed: Hybrid search failed: float division by zero
```

The same question 14 failed in every single run.

**Cause.** The crash came from two functions inside `hybrid_retriever.py`:

- `keyword_search` was written so that it returns the `top_k` indices
  regardless of their score. It therefore returned chunks that scored 0 if they
  happened to be in the `top_k`. This is a problem on its own, since we should
  not be sending chunks with no keyword match to `hybrid_search` at all, but in
  the case where every chunk scored 0 the system crashed outright.
- `hybrid_search` then normalises the vector and keyword scores. Since all the
  keyword scores were 0 for question 14, the division by zero described above
  happened.

**Fix.** Drop the zero score results at the source, and guard the
normalisation against an empty list:

```python
kept = [i for i in top_indices if scores[i] > 0]
top_indices = kept

if keyword_results:
    max_keyword = max(r.score for r in keyword_results)
    if max_keyword > 0:
        for r in keyword_results:
            r.score = (r.score / max_keyword) * keyword_weight
    else:
        keyword_results = []
```

**Result.**

| | Crash rate | Zero score chunks passed on |
|---|---|---|
| Before | 3.1% (1 of 32) | unbounded |
| After | 0% | 0 |

Across the final run this removed 69 zero score chunks from the candidate pool,
and the crash rate dropped to 0%.

The results of this fix can be seen in `mac_e5_bm25fix.xlsx`. Note that the hit
rate in that file is very different from the earlier runs, because it was
recorded after the embedding model had already been changed. The increase in
hit rate is not a result of the BM25 fix alone.

---

## 5. Problem with the Embedding Model and chunking

As you can see in `mac_baseline_v2.xlsx`, the hit rate (whether the correct
chunks were sent to the LLM or not) was still around 20%, which is far too low
for a RAG chatbot. The two things holding it there were the embedding model and
the size of the chunks that `enrich_chunk_with_context()` was producing, and
the two are the same problem: the chunker never checked what the model could
actually read.

### 5.1 Chunks exceeded the embedding model's token limit and were truncated

**Symptom.** The vector score spread was near zero on every question. The
maximum and minimum cosine similarity across 30 candidates differed in the
third decimal place, which means the embedding could not tell its candidates
apart at all.

**Cause.** Chunks were going into the embedding model over its token limit, and
everything past the limit was silently dropped. The vectors were built from the
first part of each chunk and the rest was invisible to search, which is why the
scores all looked alike.

Nothing in the code checked for this, because the chunker measured size in
words while the model measures in tokens.

**Fix.** Two places.

In the chunker, size chunks by tokens rather than words, using the embedding
model's own tokenizer, and pack sentences up to that budget. A single sentence
longer than the entire budget is split into pieces by `_split_oversized`
instead of becoming one oversized chunk. The leftover at the end of a section
is kept as its own chunk when it reaches `min_chunk_size`, and merged into the
previous chunk when it does not, so nothing is discarded.

In `ContextualEnhancer.enrich_chunk_with_context`, give every part of the
enriched chunk an explicit token budget:

| Part | Constant | Budget |
|---|---|---|
| Section title | `SECTION_TOKENS` | 20 |
| Content | `CONTENT_TOKENS` | 370 |
| Following context | `NEXT_TOKENS` | 40 |
| Document summary | `SUMMARY_TOKENS` | 50 |

`set_tokenizer` hands the enricher the same tokenizer the embedding model uses,
so `_count` measures what the model will actually see rather than approximating
it. `_clip` trims each part to its budget, and the parts are assembled in the
order Section, Content, Following, Belge, so the chunk leads with what it is
about.

Content is deliberately not clipped. If a chunk's content exceeds its budget it
is logged and passed through whole:

```
WARNING - Chunk ..._44 is 435 tokens, over the 370 token content budget
```

Clipping it would hide the problem. Logging it means the chunks that still
exceed budget are visible and countable instead of silently shortened.

The previous-context field was removed entirely. It repeated the end of the
preceding chunk, which is already indexed as its own chunk, so it spent budget
on text that was searchable anyway and diluted each chunk's embedding with
content that was not about it. Following context was kept but capped.

**Result.**

| | Chunks | Hit rate | MRR | Prompt tokens per question |
|---|---|---|---|---|
| Before | 151 | 20.7% | 0.119 | 3,379 |
| After | 689 | 41.4% | 0.296 | 1,709 |

Hit rate doubled, MRR more than doubled, and the prompt got roughly half the
size, because the retrieved chunks are now focused instead of padded. Two facts
in the question set that had existed in no chunk at all are now present.

A repeat of the same run (`after_embedding_run2.xlsx`) gave identical hit rate
and MRR, so this is not run to run noise.

### 5.2 The embedding model's window was too small

**Symptom.** Even with correctly sized chunks, hit rate sat at around 41%.

**Cause.** `all-MiniLM-L6-v2` has a 256 token window, and the enrichment has to
fit inside it too. Subtracting the section title, following context, document
summary and markers leaves roughly 140 tokens for the content itself.

140 tokens of Turkish is a fragment, not a passage. It is not enough to hold a
statement and the context that makes it findable, so facts were being split
across chunk boundaries and each half was too thin to match a question on its
own. The window was the binding constraint: no amount of better chunking gets
past it, because the limit belongs to the model, not the chunker.

**Fix.** Switched to `intfloat/multilingual-e5-small`, which has a 512 token
window. That leaves 370 tokens for content after the same enrichment, roughly
two and a half times as much, and is where the `CONTENT_TOKENS` budget above
comes from. It requires `query:` and `passage:` prefixes, which were added at
both ends.

As a side benefit, e5 is multilingual where `all-MiniLM-L6-v2` is trained on
English, which is worth something on an 85 page Turkish report. But the window
was the reason for the change.

**Result.**

| | Chunks | Hit rate | MRR | Prompt tokens per question |
|---|---|---|---|---|
| Before | 689 | 41.4% | 0.296 | 1,709 |
| After | 485 | 62.1% | 0.421 | 2,262 |

The results are in `mac_e5.xlsx`.

---

## 6. Problem in the Inclusion Rule

**Symptom.** This one was not found by a metric. It was found by reading
`hybrid_search` while trying to understand why BM25 never seemed to change the
final five.

**Cause.** After fusion, the code injects candidates into the final list to
make sure both retrieval methods are represented. Every injection wrote to the
same position:

```python
top_candidates[-1] = pool[idx]
```

So each injected candidate overwrote the previous one. If the rule decided
three candidates should be added, two were silently thrown away and only the
third survived. The list looked full and the code reported success, so nothing
anywhere signalled that the rule was doing a third of its job.

**Fix.** Walk the slot backwards so each injection lands in its own position:

```python
slot = len(top_candidates) - 1
while to_add > 0 and idx < len(pool) and slot >= 0:
    top_candidates[slot] = pool[idx]
    to_add -= 1
    idx += 1
    slot -= 1
```

**Result.**

| | Hit rate | MRR |
|---|---|---|
| Before | 60.0% | 0.407 |
| After | 56.7% | 0.390 |

The results are in `mac_e5_inclusion.xlsx`.

The measurement got worse. One question that previously landed inside the top
five no longer does.

I kept the fix anyway. The rule now does what it was written to do, and the
previous number was inflated by a bug that happened to be favourable on this
question set. A rule that inserts one candidate when it meant to insert three
is not a working rule that happens to score higher, it is a broken rule whose
breakage was helping by accident. Reverting to it would mean choosing a number
over correct behaviour, and the next document or the next question set would
not extend the same courtesy.

---

## 7. The evaluation set was under-counting

**Symptom.** Several questions came back with `hit_final = 0` while the
generated answer was correct. That combination means one of two things: the
answer is not grounded in what was retrieved, or the evidence phrase is wrong.
Checking the chunks showed it was the second one.

**Cause.** Each question accepted a single evidence phrase, but the document
states the same fact in several places and in several forms.

q03 is the clearest case. The bank member count appears in a sentence,
`64 banka, 24 tüketici finansmanı...`, and again in a year by year table,
`Banka 52 53 54 55 57 63 64`. Retrieval returns the table, the model answers 64
correctly, and the question scored zero because the evidence only accepted the
sentence.

q14 is the same problem in a different shape: Greendeks is described six
different ways across the document and the evidence accepted one of them.

A third case comes from the parser rather than the document. The PDF hyphenates
across line breaks and the parser drops the hyphen but keeps the space, so the
chunks contain `plat form`, `inşa at` and `ardın dan`. An evidence phrase
spelled correctly can never match those.

**Fix.** `evidence` became a list of accepted phrases rather than a single
string, and every phrase was verified against the chunks in the knowledge base
rather than against the source PDF. The 30 scored questions now carry 53
phrases between them.

**Result.**

| | File | Hit rate | MRR |
|---|---|---|---|
| Before | `mac_e5_inclusion.xlsx` | 56.7% | 0.390 |
| Multi-phrase evidence | `mac_e5_corrected.xlsx` | 73.3% | 0.556 |
| q03 corrected | `mac_e5_final.xlsx` | 80.0% | 0.596 |

No code changed between these three runs. The system behaved identically and
the metric stopped scoring correct retrievals as misses. The honest reading is
that retrieval was already at roughly 80% once the inclusion fix was in, and
the instrument was understating it by more than twenty points.

Every earlier number in this report carries the same understatement, since they
were measured with the original evidence set and were not re-run.

---

## 8. Final Results

Two machines, two roles. Retrieval is reported from the MacBook because the
whole before and after sequence was measured there on one answering model.
Answer accuracy is reported from the PC because llama3.2 cannot judge.

### Before and after

Machine B has a complete before and after, because both runs used a judge that
works.

| | Baseline `pc_baseline.xlsx` | Final `pc_e5_final.xlsx` |
|---|---|---|
| **Answer accuracy** | **21.9%** | **68.8%** |
| Hit rate @5 | 23.3% | 76.7% |
| MRR | 0.172 | 0.581 |
| Crash rate | 0.0% | 0.0% |
| Median latency | 18.0 s | 18.1 s |
| p90 latency | 22.8 s | 21.8 s |
| Median input tokens | 2,476 | 2,665 |

Answer accuracy roughly tripled, from just over one question in five to just
over two in three. Latency is unchanged, and the prompt grew by about 8%
because each retrieved chunk now carries more content.

Machine A shows the token side of the same changes, and the two reliability
bugs that only appeared there.

| | Baseline `mac_baseline_v2.xlsx` | Final `mac_e5_final.xlsx` |
|---|---|---|
| Hit rate @5 | 20.7% | 80.0% |
| MRR | 0.119 | 0.596 |
| Crash rate | 3.1% | 0.0% |
| Queries with NaN rerank | 100% | 0.0% |
| Mean input tokens | 3,379 | 2,278 |
| Total tokens (in + out) | 118,730 | 87,169 |
| Median latency | 26.9 s | 41.2 s |

Total token usage fell by 27%, because the retrieved chunks are focused
instead of padded with truncated filler.

Latency on machine A got worse, and the reranker is the reason. In the
baseline it returned NaN for every pair and the code fell straight through to
a constant 0.5, so reranking cost almost nothing because it was not happening.
Once it works it scores every candidate against the query, which is real
compute. The baseline was faster because it was doing less. Machine B shows no
such change, since the reranker was working there from the start.

| | Machine A (MacBook, llama3.2) | Machine B (PC, qwen2.5:7b) |
|---|---|---|
| | `mac_e5_final.xlsx` | `pc_e5_final.xlsx` |
| Hit rate @5 | **80.0%** | 76.7% |
| MRR | **0.596** | 0.581 |
| Answer accuracy | not measurable | **68.8%** |
| Crash rate | 0.0% | 0.0% |
| Queries with NaN rerank | 0.0% | 0.0% |
| Median latency | 41.2 s | 18.1 s |
| Mean input tokens | 2,278 | 2,615 |
| Total run time | 1,334 s | 590 s |

The two hit rates are close but they are not the same measurement. The pipeline
generates its search query with an LLM call before retrieval runs, so a
different answering model produces a different query and a different candidate
set. 76.7% is the same system under a different generation model, not a
reproduction of 80.0%.

Answer accuracy is reported for the final system only. No baseline was measured
with a working judge, so there is no before figure to compare it to.

### The retrieval progression

All on machine A, same answering model throughout.

| Stage | File | Hit rate | MRR |
|---|---|---|---|
| Baseline | `mac_baseline_v2.xlsx` | 20.7% | 0.119 |
| Token sized chunks | `after_embedding.xlsx` | 41.4% | 0.296 |
| multilingual-e5-small | `mac_e5.xlsx` | 62.1% | 0.421 |
| BM25 zero score fix | `mac_e5_bm25fix.xlsx` | 60.0% | 0.407 |
| Inclusion rule fix | `mac_e5_inclusion.xlsx` | 56.7% | 0.390 |
| Evidence set corrected | `mac_e5_corrected.xlsx` | 73.3% | 0.556 |
| q03 evidence corrected | `mac_e5_final.xlsx` | 80.0% | 0.596 |

The last two rows are not code changes, as described in section 7. The code
moved retrieval to roughly 57%, and correcting the evidence showed the real
figure there was already around 80%.

The denominator also changes at the BM25 fix. The first three runs exclude a
crashed question and divide by 29, the rest divide by 30, which is why 62.1%
becomes 60.0% when the only change was removing the crash.

---

## 9. Confirmed but not fixed

Found and verified, left alone for time or scope.

| Where | Finding |
|---|---|
| `_semantic_segment` | TextTiling raises `No paragraph breaks were found` on every document, because `clean_text` removes newlines first. The exception is swallowed, so the documented semantic chunking has never executed. |
| fusion | Vector and BM25 score ranges are not comparable, so the configured 70/30 weights do not mean what they say. The smallest vector contribution always exceeds the largest BM25 one, which makes vector a membership gate rather than a ranking signal. |
| `retrieve` | In hybrid mode only the LLM generated queries are searched. The user's own question reaches the searches only when query generation fails. |
| `determine_search_strategy` | Costs one LLM call per question and every decision it makes is overridden by the caller, so its output is discarded. One of the three LLM calls per question does nothing. |
| Cross encoder | Min-max normalisation means the best candidate always scores 1.0 even when every raw score is negative, so the system cannot distinguish "these five are relevant" from "these five are the least bad". |
| PDF parser | Drops the hyphen but keeps the space on words split across line breaks, so `platform` becomes `plat form`. This quietly degrades BM25 on those tokens. |
| Ingestion | Embeddings are generated one chunk at a time, and BM25 is rebuilt from the entire store after every document. |
| Cross-language retrieval | All 3 English questions miss while the identical Turkish questions hit at ranks 1 to 3. A probe on the three paired questions showed the embedding ranks the correct chunk first for the English form, and the cross-encoder then scores it negative and buries it. `ms-marco-MiniLM-L-6-v2` is trained on English query against English passage, so it has no basis for scoring a Turkish passage. A multilingual cross-encoder is the fix. |

---

## 10. Reproducing this

Everything runs locally. No paid APIs.

**Setup**

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
ollama pull qwen2.5:7b
```

Set `OLLAMA_MODEL` in `.env` to the model that should answer. This value is
machine specific, so check it before running on a second machine.

**Build the knowledge base**

Start the app, create a knowledge base using `intfloat/multilingual-e5-small`
as the embedding model, and upload `kkbfaaliyetraporu2024.pdf` to it. A correct
build reports 485 chunks.

Any change to the chunker or the embedding model needs a **new** knowledge
base, because Chroma stores the vectors as they were written and the existing
ones came from a different model.

The knowledge base id is the key in `.knowledge_bases.json`, and also the
folder name under `chroma_db/`.

**Run the evaluation**

```bash
# retrieval and answers, one CSV per run
python eval/run_eval.py --kb <kb_id> --judge-model qwen2.5:7b --out eval/results/run.csv

# retrieval only, no answer call and no judge
python eval/run_eval.py --kb <kb_id> --no-answer --out eval/results/retr.csv
```

**Read the results**

```bash
# table and summary in the terminal
python eval/report.py eval/results/run.csv

# also write the formatted workbook: Summary, Questions, Scores
python eval/report.py eval/results/run.csv --xlsx eval/results/run.xlsx

# compare two runs side by side
python eval/report.py eval/results/before.csv eval/results/after.csv --xlsx cmp.xlsx
```

Only compare runs from the same machine with the same answering model. The
pipeline generates its search query with an LLM call before retrieval runs, so
changing the answering model changes what gets retrieved.
