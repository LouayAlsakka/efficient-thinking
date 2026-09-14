#!/usr/bin/env python3
"""Turn the 70-poem train split into mlx-lm LoRA data — in the SAME shape P0 asks in.

    python3 make_lora_data.py --out lora_data/

Mechanism E trains on the task P2/P3 will measure: given a topic and a form, write the poem.
So the training example is the P0 PROMPT verbatim with the poem's own title as the topic, and
the poem as the answer. Training on a different shape than the one scored would measure the
shape change as much as the voice.

THE VALIDATION SET COMES OUT OF THE 70, and that is a cost, not a detail: mlx-lm requires a
valid.jsonl, the held-out 33 are reserved for the judge and the memorization probes, so the
only place left is the train split itself. 63 train / 7 valid, chosen by the same seed, and the
seven are NAMED in the manifest so nobody has to wonder which poems the loss curve saw.

The answer is the `text` field — 理's ruling (iii): re-segmented presentation, never `text_raw`.
"""
import argparse, hashlib, json, os, random, re

CJK = re.compile(r"[㐀-鿿]")
FORMS = {20: ("五絕", 5, 4), 28: ("七絕", 7, 4), 40: ("五律", 5, 8), 56: ("七律", 7, 8)}
PROMPT = ("請以「{topic}」為題，寫一首{form}。\n"
          "格律要求：共{n}句，每句{per}字，押平水韻，合平仄。\n"
          "只輸出詩句本身，每句一行，不要標題、不要解釋。")
PROMPT_FREE = ("請以「{topic}」為題，寫一首詩。\n"
               "只輸出詩句本身，每句一行，不要標題、不要解釋。")


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--train", default=os.path.join(here, "corpus", "tangyin_train.jsonl"))
    ap.add_argument("--out", default=os.path.join(here, "lora_data"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--valid", type=int, default=7)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.train)]
    ex, forms = [], {}
    for r in rows:
        t = "".join(CJK.findall(r["text"]))
        f = FORMS.get(len(t))
        if f:
            name, per, n = f
            p = PROMPT.format(topic=r["title"], form=name, n=n, per=per)
        else:
            name = None
            p = PROMPT_FREE.format(topic=r["title"])
        forms[name] = forms.get(name, 0) + 1
        ex.append({"messages": [{"role": "user", "content": p},
                                {"role": "assistant", "content": r["text"]}],
                   "_title": r["title"], "_form": name})

    rnd = random.Random(a.seed)
    idx = list(range(len(ex)))
    rnd.shuffle(idx)
    vi = set(idx[:a.valid])
    os.makedirs(a.out, exist_ok=True)
    for name, keep in (("train", lambda i: i not in vi), ("valid", lambda i: i in vi)):
        p = os.path.join(a.out, name + ".jsonl")
        with open(p, "w") as f:
            for i, e in enumerate(ex):
                if keep(i):
                    f.write(json.dumps({"messages": e["messages"]}, ensure_ascii=False) + "\n")
        print(f"  {name}: {sum(1 for i in range(len(ex)) if keep(i))} -> {p}")

    man = {"source": os.path.basename(a.train), "seed": a.seed,
           "n": len(ex), "n_train": len(ex) - len(vi), "n_valid": len(vi),
           "valid_titles": [ex[i]["_title"] for i in sorted(vi)],
           "forms_in_the_split": forms,
           "prompt_template": PROMPT, "prompt_template_unregulated": PROMPT_FREE,
           "answer_field": "text (re-segmented) — 理's ruling (iii); text_raw is never trained on",
           "why_valid_comes_out_of_train": "mlx-lm requires valid.jsonl; the held-out 33 are "
                                           "reserved for the judge and the P4 probes, so the only "
                                           "place left is the train split. It costs 7 of 70.",
           "thinness": "63 training examples. This is thin and it belongs in the row, not only in "
                       "the prose: any P2/P3 effect has to be read against it."}
    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), ensure_ascii=False, indent=1)
    print(f"  forms: {forms}")
    print(f"  valid held from train: {man['valid_titles']}")


if __name__ == "__main__":
    main()
