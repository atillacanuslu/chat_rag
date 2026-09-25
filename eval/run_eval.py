"""Evaluation runner for the RAG pipeline.

One script, one CSV. Two modes:

    full (default)   pipeline.query()  -> retrieval metrics + answer + judge verdict
    --no-answer      pipeline.retrieve() -> retrieval metrics only, no answer, no judge
                     (faster; use while tuning chunking, fusion or the reranker)

Checks performed:
    check 1  evidence phrase found in the final chunks   -> string matching, no LLM
    check 2  answer graded by a judge model              -> one plain LLM call,
                                                            made by this script,
                                                            outside the pipeline

Usage:
    python eval/run_eval.py --list-kb
    python eval/run_eval.py --kb 9a0d791f --out eval/results/baseline.csv
    python eval/run_eval.py --kb 9a0d791f --no-answer --out eval/results/retr.csv
    python eval/run_eval.py --kb 9a0d791f --limit 5 --verbose
"""

import argparse
import csv
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.normalize import contains

JUDGE_FACT = """Soru: {question}
Beklenen cevap: {expected}
Verilen cevap: {given}

Verilen cevap, beklenen cevabi dogru sekilde iceriyor mu?
Farkli yazim veya bicim (11 Nisan 1995 ile 11.04.1995 gibi) ayni sayilir.
Yanlis bir deger verilmisse veya cevap beklenen bilgiyi icermiyorsa YANLIS yaz.
Sadece tek kelime yaz: DOGRU veya YANLIS"""

JUDGE_NO_ANSWER = """Soru: {question}
Verilen cevap: {given}

Verilen cevap, bu bilginin belgede bulunmadigini veya cevabin bilinmedigini soyluyor mu?
Cevap bir bilgi uyduruyorsa HAYIR yaz.
Sadece tek kelime yaz: EVET veya HAYIR"""


# ----------------------------------------------------------------- helpers
def load_questions(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_kb_registry(repo_root):
    path = os.path.join(repo_root, ".knowledge_bases.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("knowledge_bases", data)


def build_pipeline(kb_id, repo_root):
    """Build a pipeline for one knowledge base, mirroring app.py's settings override."""
    from config.settings import Settings
    from pipeline.rag_pipeline import RAGPipeline

    settings = Settings()
    settings.enable_conversation = False          # each question must be independent

    if kb_id:
        kb = load_kb_registry(repo_root).get(kb_id)
        if not kb:
            raise SystemExit("Knowledge base '%s' not found in .knowledge_bases.json" % kb_id)
        settings.embedding_model_name = kb.get("embedding_model_name", settings.embedding_model_name)
        settings.vector_db_provider = kb.get("vector_db_provider", "chroma")
        settings.vector_db_path = kb.get("vector_db_path") or "./chroma_db/%s" % kb_id
        if kb.get("chunker"):
            settings.kb_chunker_config = kb["chunker"]

    return RAGPipeline(settings=settings)


def span(values):
    if not values:
        return "", "", ""
    return round(min(values), 4), round(max(values), 4), round(max(values) - min(values), 4)


def judge(judge_model, question, expected, given, no_answer=False):
    """One plain LLM call, outside the pipeline. Returns (correct, raw_verdict)."""
    import ollama

    if no_answer:
        prompt = JUDGE_NO_ANSWER.format(question=question, given=given)
        positive = "EVET"
    else:
        prompt = JUDGE_FACT.format(question=question, expected=expected, given=given)
        positive = "DOGRU"

    try:
        resp = ollama.chat(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0, "num_predict": 300},
        )
        verdict = (resp.get("message", {}) or {}).get("content", "").strip().upper()
    except Exception as exc:
        return None, "JUDGE_ERROR: %s" % exc

    clean = verdict.replace("\u011e", "G").replace("\u0130", "I").replace("\u015e", "S")
    if positive in clean:
        return True, verdict[:60]
    if ("YANLIS" in clean) or ("HAYIR" in clean):
        return False, verdict[:60]
    return None, verdict[:60]          # unclear, review by hand


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default=None, help="knowledge base id from .knowledge_bases.json")
    ap.add_argument("--questions", default="eval/questions.jsonl")
    ap.add_argument("--out", default="eval/results/run.csv")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--method", default="hybrid", choices=["hybrid", "vector", "bm25", "auto"],
                    help="'auto' passes None so the pipeline's strategy call picks the method")
    ap.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", "llama3.2:latest"),
                    help="keep this identical across every run")
    ap.add_argument("--no-answer", action="store_true",
                    help="retrieval only: skip the answer call and the judge")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verbose", action="store_true", help="print the score trail per question")
    ap.add_argument("--xlsx", default=None,
                    help="also write a formatted workbook (defaults to the CSV path with .xlsx)")
    ap.add_argument("--list-kb", action="store_true")
    args = ap.parse_args()

    method = None if args.method == "auto" else args.method

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if args.list_kb:
        for kid, kb in load_kb_registry(repo_root).items():
            print("%s  %s  %s  %s" % (kid, kb.get("name", ""), kb.get("vector_db_provider"),
                                      kb.get("embedding_model_name")))
        return

    questions = load_questions(args.questions)
    if args.limit:
        questions = questions[: args.limit]
    mode = "retrieval only" if args.no_answer else "full (answer + judge)"
    print("Loaded %d questions. Mode: %s" % (len(questions), mode))

    pipeline = build_pipeline(args.kb, repo_root)
    print("Pipeline ready. chunks in store: %s" % pipeline.vector_db.count())

    llm = pipeline.llm_model
    has_usage = hasattr(llm, "get_usage")          # exists only after instrumentation

    # warm up: the first call loads models and builds BM25
    try:
        if args.no_answer:
            pipeline.retrieve("isinma sorgusu", top_k=args.top_k, retrieval_method=method,
                              use_reranking=True)
        else:
            pipeline.query("isinma sorgusu", top_k=args.top_k, retrieval_method=method)
    except Exception:
        pass

    rows = []
    for q in questions:
        no_answer_q = q["type"] == "no_answer"
        # evidence may be a single phrase or a list of accepted phrases. A fact is
        # often stated in several places in the document, so pinning one wording
        # undercounts retrieval: the system finds a different, equally valid
        # statement of the same fact and it scores as a miss.
        ev = q.get("evidence") or []
        evidence_list = [ev] if isinstance(ev, str) else list(ev)
        scored = bool(evidence_list) and not no_answer_q
        if has_usage:
            llm.reset_usage()

        answer, results, error = "", [], ""
        t0 = time.perf_counter()
        try:
            if args.no_answer:
                results, metadata = pipeline.retrieve(
                    query=q["question"], top_k=args.top_k,
                    use_reranking=True, retrieval_method=method)
            else:
                out = pipeline.query(
                    question=q["question"], top_k=args.top_k,
                    use_reranking=True, retrieval_method=method,
                    temperature=0.3, max_tokens=500)
                answer = out.get("answer", "")
                metadata = out.get("metadata", {}) or {}
                results = metadata.get("final_results", [])
                if not results:
                    # fall back to the full chunk texts if the pipeline exposes them
                    contents = metadata.get("final_contents", [])
                    results = [_Shim(c) for c in contents]
        except Exception as exc:
            metadata = {}
            error = "%s: %s" % (type(exc).__name__, exc)
        elapsed = time.perf_counter() - t0

        # ---- check 1: string matching, no LLM
        rank = 0
        matched_evidence = ""
        if scored and results:
            for i, r in enumerate(results, 1):
                text = getattr(r, "content", None) or r.chunk.content
                hit = next((e for e in evidence_list if contains(text, e)), None)
                if hit:
                    rank = i
                    matched_evidence = hit
                    break

        final_scores = [r.score for r in results if hasattr(r, "score") and isinstance(r.score, (int, float))]
        raw_scores = [s for s in (getattr(r, "raw_score", None) for r in results) if isinstance(s, (int, float))]
        f_min, f_max, f_spread = span(final_scores)
        r_min, r_max, r_spread = span(raw_scores)

        # per stage score diagnostics (populated once the instrumentation patches are applied)
        stats = (metadata or {}).get("stage_stats", {}) or {}
        retr = stats.get("retriever", {}) or {}
        vec = retr.get("vector", {}) or {}
        bm = retr.get("bm25", {}) or {}
        hyb = retr.get("hybrid", {}) or {}
        rr_stats = stats.get("reranker", {}) or {}

        def mm(d, lo="min", hi="max"):
            a, b = d.get(lo), d.get(hi)
            if a is None or b is None:
                return "", "", ""
            return round(a, 4), round(b, 4), round(b - a, 4)

        v_min, v_max, v_spread = mm(vec)
        b_min, b_max, b_spread = mm(bm)
        h_min, h_max, h_spread = mm(hyb)
        rk_min, rk_max, rk_spread = mm(rr_stats, "raw_min", "raw_max")

        # ---- check 2: judge (skipped in --no-answer mode)
        correct, verdict = "", ""
        if not args.no_answer:
            if error:
                correct, verdict = 0, "PIPELINE_ERROR"
            else:
                c, verdict = judge(args.judge_model, q["question"], q.get("answer", ""),
                                   answer, no_answer=no_answer_q)
                correct = "" if c is None else int(c)

        rows.append({
            "id": q["id"], "type": q["type"], "scored": int(scored),
            "question": q["question"],
            "expected": q.get("answer", ""),
            "evidence": " | ".join(evidence_list),
            "matched_evidence": matched_evidence,
            "hit_final": int(rank > 0) if scored else "",
            "rank": rank, "rr": round(1 / rank, 4) if rank else 0.0,
            "correct": correct, "verdict": verdict,
            "n_results": len(results),
            "final_min": f_min, "final_max": f_max, "final_spread": f_spread,
            "raw_min": r_min, "raw_max": r_max, "raw_spread": r_spread,
            "tie_fallback": int(f_spread == 0 and len(results) > 1) if final_scores else "",
            "vec_n": vec.get("n", ""), "vec_min": v_min, "vec_max": v_max, "vec_spread": v_spread,
            "bm25_n": bm.get("n", ""), "bm25_min": b_min, "bm25_max": b_max,
            "bm25_spread": b_spread, "bm25_zero_count": bm.get("zero_count", ""),
            "hybrid_n": hyb.get("n", ""), "hybrid_min": h_min, "hybrid_max": h_max, "hybrid_spread": h_spread,
            "rerank_n": rr_stats.get("n", ""), "rerank_raw_min": rk_min, "rerank_raw_max": rk_max,
            "rerank_raw_spread": rk_spread, "rerank_nan_count": rr_stats.get("nan_count", ""),
            "rerank_empty_chunks": rr_stats.get("empty_chunks", ""),
            "vec_scores": json.dumps(vec.get("scores", [])),
            "bm25_scores": json.dumps(bm.get("scores", [])),
            "hybrid_scores": json.dumps(hyb.get("scores", [])),
            "hybrid_sources": json.dumps(hyb.get("sources", [])),
            "rerank_raw_scores": json.dumps(rr_stats.get("raw_scores", [])),
            "seconds": round(elapsed, 2),
            "prompt_tokens": (llm.get_usage().get("prompt_tokens", "") if has_usage and not error else ""),
            "completion_tokens": (llm.get_usage().get("completion_tokens", "") if has_usage and not error else ""),
            "llm_calls": (llm.get_usage().get("call_count", "") if has_usage and not error else ""),
            "error": error,
            "answer": answer.replace("\n", " ")[:400],
        })

        if error:
            mark = "ERROR"
        elif args.no_answer:
            mark = "hit  " if rank else "miss "
        else:
            mark = {1: "ok   ", 0: "WRONG", "": "?    "}[correct]
        print("%s  %s hit=%s rank=%-3s %6.1fs  %s" % (
            q["id"], mark, rows[-1]["hit_final"], rank or "-", elapsed, q["question"][:44]))
        if not args.no_answer and answer:
            print("        -> %s" % answer[:110])
        if args.verbose and results:
            for i, r in enumerate(results, 1):
                raw = getattr(r, "raw_score", None)
                sc = getattr(r, "score", "")
                print("        %d. %s%s  %s" % (
                    i, ("%.3f" % sc) if sc != "" else "",
                    (" raw=%.3f" % raw) if raw is not None else "",
                    getattr(getattr(r, "chunk", None), "section_title", "")))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- readable output
    try:
        from eval.report import summarize, print_terminal, write_xlsx
    except ImportError:
        from report import summarize, print_terminal, write_xlsx

    str_rows = [{k: ("" if v is None else v) for k, v in r.items()} for r in rows]
    for r in str_rows:
        for k, v in list(r.items()):
            r[k] = str(v) if not isinstance(v, str) else v

    summary = summarize(str_rows)
    print_terminal([args.out], [str_rows], [summary])

    xlsx_path = args.xlsx
    if xlsx_path is None and args.out.endswith(".csv"):
        xlsx_path = args.out[:-4] + ".xlsx"
    if xlsx_path:
        try:
            write_xlsx([args.out], [str_rows], [summary], xlsx_path)
        except Exception as exc:
            print("could not write workbook: %s" % exc)

    print("raw results written to %s" % args.out)


class _Shim(object):
    """Wraps a plain chunk string so the check-1 loop can read .content."""
    def __init__(self, content):
        self.content = content
        self.score = ""


if __name__ == "__main__":
    main()