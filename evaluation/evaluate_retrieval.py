import sys
import os
import json
from pathlib import Path

# -------- Project root import --------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rag.pipeline import RAGPipeline


BASE_DIR = Path(__file__).resolve().parent
TEST_DIR = BASE_DIR / "test"
RESULTS_DIR = BASE_DIR / "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# Load ALL batches
def load_testset():
    testset = []

    for file in sorted(TEST_DIR.glob("test_set_batch_*.json")):
        with open(file, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
            testset.extend(data)

    print(f"Loaded {len(testset)} questions")
    return testset


# Metrics
def strict_hit(retrieved, gt):
    """At least 1 exact GT match"""
    if not gt:
        return int(len(retrieved) == 0)
    return int(any(c in gt for c in retrieved))


def relaxed_hit(retrieved, gt, min_overlap):
    """At least N GT matches"""
    if not gt:
        return int(len(retrieved) == 0)

    overlap = len(set(retrieved) & set(gt))
    return int(overlap >= min_overlap)


def recall_at_k(retrieved, gt):
    if not gt:
        return int(len(retrieved) == 0)

    return len(set(retrieved) & set(gt)) / len(gt)


def precision_at_k(retrieved, gt):
    if not retrieved:
        return 0

    if not gt:
        return int(len(retrieved) == 0)

    return len(set(retrieved) & set(gt)) / len(retrieved)


def reciprocal_rank(retrieved, gt):
    if not gt:
        return int(len(retrieved) == 0)

    for i, cid in enumerate(retrieved, start=1):
        if cid in gt:
            return 1 / i
    return 0


# Evaluation
def evaluate():
    pipeline = RAGPipeline(top_k=10)
    testset = load_testset()

    total_strict = 0
    total_relaxed = 0
    total_recall = 0
    total_precision = 0
    total_mrr = 0

    empty_total = 0
    empty_correct = 0

    raw_results = []

    for item in testset:
        qid = item["id"]
        query = item["question"]
        gt = item.get("expected_chunks", [])
        min_overlap = item.get("expected_min_overlap", 1)

        # -------- PURE RETRIEVAL --------
        retrieved_chunks = pipeline.retrieve(query, use_memory=False)

        retrieved_ids = []
        for chunk in retrieved_chunks:
            cid = chunk.get("chunk_id", -1)
            if cid is not None and cid >= 0:
                retrieved_ids.append(cid)

        retrieved_ids = retrieved_ids[:10]

        overlap = list(set(retrieved_ids) & set(gt))

        # -------- EMPTY GT HANDLING --------
        if not gt:
            empty_total += 1
            if len(retrieved_ids) == 0:
                empty_correct += 1

        # -------- METRICS --------
        s = strict_hit(retrieved_ids, gt)
        rlx = relaxed_hit(retrieved_ids, gt, min_overlap)
        r = recall_at_k(retrieved_ids, gt)
        p = precision_at_k(retrieved_ids, gt)
        m = reciprocal_rank(retrieved_ids, gt)

        total_strict += s
        total_relaxed += rlx
        total_recall += r
        total_precision += p
        total_mrr += m

        print(f"Q{qid} | Strict: {s} | Relaxed: {rlx} | Overlap: {overlap}")

        raw_results.append({
            "id": qid,
            "query": query,
            "gt": gt,
            "retrieved": retrieved_ids,
            "overlap": overlap,
            "strict_hit": s,
            "relaxed_hit": rlx,
            "recall": round(r, 3),
            "precision": round(p, 3),
            "mrr": round(m, 3)
        })

    # FINAL METRICS
    n = len(testset)

    print("\n===== RETRIEVAL METRICS =====")
    print(f"Questions: {n}")
    print(f"Strict Hit@10: {total_strict / n:.3f}")
    print(f"Relaxed Hit@10: {total_relaxed / n:.3f}")
    print(f"Recall@10: {total_recall / n:.3f}")
    print(f"Precision@10: {total_precision / n:.3f}")
    print(f"MRR: {total_mrr / n:.3f}")

    if empty_total > 0:
        print(f"Unanswerable Accuracy: {empty_correct}/{empty_total} = {empty_correct / empty_total:.3f}")

    # SAVE RESULTS
    out_file = RESULTS_DIR / "retrieval_results.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(raw_results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved results to: {out_file}")


if __name__ == "__main__":
    evaluate()