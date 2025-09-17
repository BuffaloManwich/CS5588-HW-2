import sys, json, hashlib

ALIASES = {
    "image":       ["image","img","image_path","path","image_file","img_path"],
    "text":        ["text","question","prompt","query","instruction"],
    "question_id": ["question_id","qid","id","uid","example_id"],
    "category":    ["category","task","type","class"],
    "answer_text": ["answer_text","answer","label","gt","ground_truth"],
}
def pick(d, keys, default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default

def normalize_line(raw):
    out = {
        "image":       pick(raw, ALIASES["image"]),
        "text":        pick(raw, ALIASES["text"]),
        "question_id": pick(raw, ALIASES["question_id"]),
        "category":    pick(raw, ALIASES["category"], "open"),
        "answer_text": pick(raw, ALIASES["answer_text"], ""),
    }
    if not out["question_id"]:
        t = (out.get("image") or "") + "|" + (out.get("text") or "")
        out["question_id"] = "auto_" + str(abs(hash(t)))
    return out
def valid(sample):
    return all([
        sample.get("image"),
        sample.get("text"),
        sample.get("question_id"),
        sample.get("category") is not None,
        sample.get("answer_text") is not None
    ])
def main(inp, outp):
    n_in = n_out = n_bad = 0
    with open(inp, "r") as fin, open(outp, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                n_bad += 1
                continue
            n_in += 1
            sample = normalize_line(obj)
            if valid(sample):
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                n_out += 1
            else:
                n_bad += 1
    print(json.dumps({"read": n_in, "written": n_out, "skipped": n_bad}, indent=2))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python reproduce/prepare_eval_file.py IN.jsonl OUT.jsonl", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
