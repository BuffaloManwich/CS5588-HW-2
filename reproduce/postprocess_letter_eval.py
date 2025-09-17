import json, sys, re

"""
Rescore predictions by extracting the first letter A–E from the model output.
Inputs:
  1) predictions.json  (dict: {question_id: {"pred": "...", ...}, ...})
  2) eval_jsonl        (schema: image, text, question_id, category, answer_text)
Outputs:
  - prints overall accuracy
  - writes a small report JSON next to the predictions file (same prefix + ".rescored.json")
"""

LETTER_RE = re.compile(r'\b([A-E])\b')
NO_CORRECT_RE = re.compile(r'no\s+correct', re.I)

def gold_letter(cat: str) -> str:
    # category like "D." -> "D"
    if not cat:
        return ""
    m = re.match(r'\s*([A-E])\s*\.?', cat)
    return m.group(1) if m else ""

def pred_letter(text: str) -> str:
    if not text:
        return ""
    # if it explicitly says no correct, interpret as E
    if NO_CORRECT_RE.search(text):
        return "E"
    m = LETTER_RE.search(text)
    return m.group(1) if m else ""

def load_eval(eval_path):
    gold = {}
    with open(eval_path, "r") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            qid = obj.get("question_id")
            cat = obj.get("category", "")
            if qid:
                gold[qid] = gold_letter(cat)
    return gold

def main(pred_path, eval_path):
    with open(pred_path, "r") as f:
        preds = json.load(f)

    gold = load_eval(eval_path)

    total = correct = 0
    rows = []
    for qid, rec in preds.items():
        raw = rec.get("pred") if isinstance(rec, dict) else str(rec)
        p = pred_letter(str(raw))
        g = gold.get(qid, "")
        ok = (p == g) and (p != "")
        total += 1
        correct += int(ok)
        rows.append({"question_id": qid, "pred_raw": raw, "pred_letter": p, "gold": g, "ok": ok})

    acc = (correct / total) * 100 if total else 0.0
    report = {"total": total, "correct": correct, "acc": acc, "items": rows}

    out_path = pred_path.replace(".json", ".rescored.json")
    with open(out_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Rescored ACC: {acc:.2f}% ({correct}/{total})")
    print(f"Wrote: {out_path}")
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python reproduce/postprocess_letter_eval.py PRED_JSON EVAL_JSONL", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
