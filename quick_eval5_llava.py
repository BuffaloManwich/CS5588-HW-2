#!/usr/bin/env python3
import os, json, sys, re
from PIL import Image
import torch
import numpy as np
from transformers import AutoTokenizer, AutoImageProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from transformers.generation import LogitsProcessor, LogitsProcessorList

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
DTYPE = torch.float16  # for 4-bit compute
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LETTER_RE = re.compile(r"\b([A-E])\b", re.IGNORECASE)

def parse_options(question_text: str):
    if not isinstance(question_text, str): return {}
    m = re.search(r"Answers:\s*(.+?)(?:\n\s*\n|\Z)", question_text, re.S|re.I)
    if not m: return {}
    block = m.group(1); opts = {}
    for L in ["A","B","C","D","E"]:
        mm = re.search(rf"(?m)^{L}\.\s*(.*?)(?=^[A-E]\.|$)", block, re.S)
        if mm: opts[L] = mm.group(1).strip()
    return opts

def gold_letter_from(example: dict, opts: dict):
    cat = example.get("category")
    if isinstance(cat, str):
        m = re.match(r"\s*([A-E])\.", cat.strip(), re.I)
        if m: return m.group(1).upper()
    gold_text = (example.get("answer_text") or "").strip()
    if gold_text and opts:
        for L,t in opts.items():
            if t.strip().lower() == gold_text.lower(): return L
    return ""

class ProcWrapper:
    def __init__(self, model_id: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
        self.image_processor = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)
    def __call__(self, text=None, images=None, return_tensors="pt"):
        out = {}
        if text is not None:   out.update(self.tokenizer(text, return_tensors="pt"))
        if images is not None: out.update(self.image_processor(images, return_tensors="pt"))
        return out

def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    return LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True,
    )

def _allowed_letter_ids(tok):
    cand = ["A","B","C","D","E"," A"," B"," C"," D"," E"]
    ids = set()
    for t in cand:
        enc = tok(t, add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()
        if enc: ids.add(enc[-1])
    return sorted(ids)

class _RestrictToIds(LogitsProcessor):
    def __init__(self, allowed_ids):
        self.allowed = sorted(set(allowed_ids)); self.mask=None; self.vocab=None
    def __call__(self, input_ids, scores):
        import torch
        V = scores.shape[-1]
        if (self.mask is None) or (self.vocab != V):
            self.vocab = V
            mask = torch.full_like(scores, float("-inf"))
            mask[..., self.allowed] = 0.0
            self.mask = mask
        return scores + self.mask

def main(path):
    torch.set_grad_enabled(False)
    proc = ProcWrapper(MODEL_ID)
    model = load_model()
    total = hits = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            img_path = ex["image"]
            q = ex["text"]
            gold_text = ex.get("answer_text", "")
            opts = parse_options(q)
            gold_letter = gold_letter_from(ex, opts)

            # Load image
            image = Image.open(img_path).convert("RGB")

            # Encode (keep tensors on CPU; accelerate will move as needed)
           
            choices = parse_options(q)
            choices_str = ""
            if choices:
                order = [k for k in ["A","B","C","D","E"] if k in choices and choices[k]]
                if order:
                    choices_str = "Options:\n" + "\n".join(f"{k}. {choices[k]}" for k in order)

            chat = (
                "You are a careful visual MCQ solver. Output ONLY the single letter A, B, C, D, or E.\n"
                "<image>\n"
                f"{q}\n"
                f"{choices_str}\n"
                "ASSISTANT:"
            )
            
            inputs = proc(text=[chat], images=[image], return_tensors="pt")
            dev = next(model.parameters()).device
            inputs = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inputs.items()}
            
            allowed = _allowed_letter_ids(proc.tokenizer)
            processors = LogitsProcessorList([_RestrictToIds(allowed)])

            # Inference with cache disabled to shrink memory
            try:
                with torch.inference_mode():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=1,
                        do_sample=False,
                        use_cache=False,  # critical for tight VRAM
                        eos_token_id=getattr(proc.tokenizer, "eos_token_id", None),
                        logits_processor=processors,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(json.dumps({
                    "qid": ex.get("question_id"),
                    "error": "CUDA OOM on generate"
                }))
                continue

            seq = out.sequences
            pred = proc.tokenizer.batch_decode(seq, skip_special_tokens=True)[0]
            if "ASSISTANT:" in pred:
                pred = pred.split("ASSISTANT:")[-1]

            pm = LETTER_RE.search(pred)
            pred_letter = pm.group(1).upper() if pm else pred.strip()[:1].upper()

            # letter probabilities for debugging
            try:
                probs = out.scores[0].softmax(-1)[0].to("cpu")
                letter_probs = {}
                for L in ["A","B","C","D","E"]:
                    ids_for_L = []
                    for t in (L, " "+L):
                        enc = proc.tokenizer(t, add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()
                        if enc: ids_for_L.append(enc[-1])
                    if ids_for_L:
                        pid = max(ids_for_L, key=lambda i: float(probs[i]))
                        letter_probs[L] = float(probs[pid])
            except Exception:
                letter_probs = {}

            ok = (pred_letter == gold_letter) if gold_letter else False
            hits += int(ok); total += 1

            print(json.dumps({
                "qid": ex.get("question_id"),
                "category": ex.get("category"),
                "gold_text": gold_text,
                "gold_letter": gold_letter,
                "pred": pred.strip(),
                "picked": pred_letter,
                "letter_probs": letter_probs,
                "match": bool(ok)
            }, ensure_ascii=False))

            # free any transient VRAM
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    acc = hits / total if total else 0.0
    print(f"\nSAMPLES={total}  ACC={acc:.3f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python quick_eval5_llava.py <data.jsonl>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
