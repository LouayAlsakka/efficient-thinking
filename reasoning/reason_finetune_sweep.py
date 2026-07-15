#!/usr/bin/env python
"""The DATA lever in the LLM domain — accuracy vs TRAINING-DATA SIZE (fine-tuning sweep).

ET-II varies model *size* and *search* for reasoning, but never *training-data size* directly (Connect-4
carries the data-size result). This closes that gap: LoRA-fine-tune one fixed small model (Qwen2.5-0.5B)
on an increasing number of GSM8K training examples, holding the OPTIMIZATION budget fixed (same iters,
same batch), so the only thing that varies is how much *unique* data the evaluator saw. Then read greedy
accuracy on a held-out test set vs data size — the reasoning analog of Connect-4's open-loop-ceiling-vs-
labels curve, and of Paper I's chess data-fraction study.

Runs entirely on llm2 (mlx_lm). Fixed compute per point isolates data from optimization:
  ./.venv/bin/python reasoning/reason_finetune_sweep.py --sizes 0,64,256,1024,4096 --iters 400 --eval 150
"""
import argparse, json, os, re, subprocess, sys, urllib.request

GSM_TRAIN_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
INSTR = "\nThink step by step, then end with: #### <number>"
BASE_PROMPT = "Question: {q}\nAnswer:"   # plain-text scaffold for a non-instruct base model (no chat template)


def gold(ans):
    return ans.split("####")[-1].strip().replace(",", "")


def extract(text):
    m = re.findall(r"####\s*(-?[0-9][0-9,]*)", text or "")
    if m:
        return m[-1].replace(",", "")
    nums = re.findall(r"-?\d[\d,]*", text or "")
    return nums[-1].replace(",", "") if nums else None


def load_train(path="reasoning/data/gsm8k_train.jsonl"):
    if not os.path.exists(path):
        print(f"[ft] downloading GSM8K train → {path}", flush=True)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(GSM_TRAIN_URL, path)
    return [json.loads(l) for l in open(path)]


def clean_answer(ans):
    """Strip GSM8K's <<calc=annotations>> surface noise; keep the reasoning text and the #### answer."""
    return re.sub(r"<<[^>]*>>", "", ans)


def write_lora_data(rows, ddir, base=False):
    """mlx_lm lora expects <ddir>/train.jsonl + valid.jsonl. Chat format for instruct models,
    plain prompt/completion for a base (non-instruct) model."""
    os.makedirs(ddir, exist_ok=True)
    def dump(rs, fn):
        with open(os.path.join(ddir, fn), "w") as f:
            for r in rs:
                if base:
                    rec = {"prompt": BASE_PROMPT.format(q=r["question"]),
                           "completion": " " + clean_answer(r["answer"])}
                else:
                    rec = {"messages": [{"role": "user", "content": r["question"] + INSTR},
                                        {"role": "assistant", "content": clean_answer(r["answer"])}]}
                f.write(json.dumps(rec) + "\n")
    n_val = max(4, len(rows) // 20)
    dump(rows[n_val:], "train.jsonl")
    dump(rows[:n_val], "valid.jsonl")


def finetune(model, ddir, adir, iters, batch, lr, layers):
    os.makedirs(adir, exist_ok=True)
    cmd = [sys.executable, "-m", "mlx_lm", "lora", "--model", model, "--train",
           "--data", ddir, "--iters", str(iters), "--batch-size", str(batch),
           "--num-layers", str(layers), "--learning-rate", str(lr),
           "--adapter-path", adir, "--steps-per-report", "50", "--val-batches", "4"]
    print("[ft] " + " ".join(cmd[3:]), flush=True)
    subprocess.run(cmd, check=True)


def evaluate(model, adir, tests, max_tokens, base=False):
    from mlx_lm import load, generate as gen
    m, tok = load(model, adapter_path=adir) if adir else load(model)
    correct = 0
    for i, p in enumerate(tests):
        if base:
            pr = tok.encode(BASE_PROMPT.format(q=p["question"]))
        else:
            pr = tok.apply_chat_template([{"role": "user", "content": p["question"] + INSTR}],
                                         add_generation_prompt=True)
        out = gen(m, tok, prompt=pr, max_tokens=max_tokens, verbose=False)   # greedy (default)
        correct += (extract(out) == gold(p["answer"]))
        if (i + 1) % 50 == 0:
            print(f"    eval {i+1}/{len(tests)}  running acc {100*correct/(i+1):.1f}%", flush=True)
    return round(100 * correct / len(tests), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--base", action="store_true",
                    help="fine-tune a non-instruct BASE model (plain prompt/completion, no chat template) "
                         "— the clean 'climb from a low floor' curve without warm-start erosion")
    ap.add_argument("--sizes", default="0,64,256,1024,4096")
    ap.add_argument("--epochs", type=float, default=3.0)   # fixed EPOCHS → more data does more real training
    ap.add_argument("--iters", type=int, default=0)        # >0 overrides epochs with a fixed iter count
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.0)       # 0 → auto: 1e-4 base (needs to learn) / 5e-5 instruct
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--eval", type=int, default=150)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reasoning/finetune_sweep_results.json")
    a = ap.parse_args()
    if a.model is None:
        a.model = "mlx-community/Qwen2.5-0.5B-4bit" if a.base else "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    if a.lr == 0.0:
        a.lr = 1e-4 if a.base else 5e-5

    import random
    train = load_train(); random.Random(a.seed).shuffle(train)
    tests = [json.loads(l) for l in open("reasoning/data/gsm8k_test.jsonl")]
    random.Random(a.seed + 1).shuffle(tests); tests = tests[:a.eval]
    sizes = [int(s) for s in a.sizes.split(",")]

    budget = f"{a.iters} fixed iters" if a.iters > 0 else f"{a.epochs} epochs"
    print(f"[ft] model={a.model} | {budget} @ batch {a.batch}, lr {a.lr} | cleaned targets | "
          f"eval on {len(tests)} held-out GSM8K", flush=True)
    curve = []
    for N in sizes:
        if N == 0:
            acc = evaluate(a.model, None, tests, a.max_tokens, a.base)   # base model, no fine-tune
            print(f"[ft] N=0 (base, no fine-tune)  acc={acc}%", flush=True)
        else:
            ddir = f"reasoning/ft_data_{N}"; adir = f"reasoning/ft_adapter_{N}"
            write_lora_data(train[:N], ddir, a.base)
            iters = a.iters if a.iters > 0 else max(20, int(a.epochs * N / a.batch))
            print(f"[ft] N={N}: {iters} iters ≈ {a.epochs} epochs @ batch {a.batch}, lr {a.lr}", flush=True)
            finetune(a.model, ddir, adir, iters, a.batch, a.lr, a.layers)
            acc = evaluate(a.model, adir, tests, a.max_tokens, a.base)
            print(f"[ft] N={N} training examples  acc={acc}%", flush=True)
        curve.append({"train_size": N, "accuracy": acc})
        json.dump({"model": a.model, "iters": a.iters, "eval_n": len(tests), "curve": curve},
                  open(a.out, "w"), indent=2)
    print(f"\n[ft] wrote {a.out}", flush=True)
    print("Read: accuracy rising with train_size at FIXED optimization budget = the data lever; where it "
          "flattens = data stops binding (capacity/base then binds).", flush=True)


if __name__ == "__main__":
    main()
