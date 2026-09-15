#!/usr/bin/env python3
"""P0: does the untrained Qwen-7B write a regulated poem at all?

    python3 p0_baseline.py --out results/p0_base_7b.jsonl --report results/p0_base_7b.json

50 topics × 4 forms = 200 prompts, temp 0.7, Qwen2.5-7B-Instruct bf16 under mlx-lm on llm1.
The prediction being scored is 理's P0: the baseline passes the form verifier on < 40% of
七絕 attempts.

THE PARSER IS PART OF THE MEASUREMENT, so it reports itself
----------------------------------------------------------
The model answers in prose with a poem inside it. Extracting the poem is a step that can fail,
and a parser that extracts nothing hands the verifier an empty string, which fails every rule
and looks exactly like a model that cannot write a 七絕. So every generation lands in exactly
one of three buckets and all three are published:

    EXTRACTED      a candidate poem of the right character count was found
    WRONG_LENGTH   CJK was found but not the count the form requires — a real model failure
    NO_POEM        nothing extractable — a PARSER result, not a model result

A pass rate quoted over EXTRACTED only, without the other two counts beside it, would be a
rate over the subset the parser happened to understand.
"""
import argparse, json, os, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CJK = re.compile(r"[㐀-鿿]")
LATIN = re.compile(r"[A-Za-z]")

TOPICS = [
    "桃花", "春雨", "醉酒", "落花", "明月", "秋風", "江南", "漁舟", "山寺", "古松",
    "梅花", "白髮", "琴聲", "竹影", "夜雨", "孤舟", "殘雪", "燕子", "美人", "離別",
    "書齋", "登高", "菊花", "寒夜", "客愁", "故鄉", "柳絮", "荷花", "螢火", "清明",
    "重陽", "除夕", "題畫", "送友", "貧居", "賣畫", "夢醒", "野渡", "牧童", "古城",
    "茶煙", "松風", "石橋", "漁火", "晚鐘", "春困", "雁聲", "苔痕", "藥爐", "閒居",
]
_BAD = [t for t in TOPICS if not re.fullmatch(r"[\u3400-\u9fff]+", t)]
assert not _BAD, f"topic list has non-CJK entries: {_BAD}"
assert len(TOPICS) == 50 and len(set(TOPICS)) == 50, f"{len(TOPICS)} topics, {len(set(TOPICS))} unique"

FORMS = [("五絕", 5, 4), ("七絕", 7, 4), ("五律", 5, 8), ("七律", 7, 8)]

PROMPT = ("請以「{topic}」為題，寫一首{form}。\n"
          "格律要求：共{n}句，每句{per}字，押平水韻，合平仄。\n"
          "只輸出詩句本身，每句一行，不要標題、不要解釋。")


def extract(text, per, n):
    """Return (poem, bucket). A poem is n hemistichs of `per` CJK characters."""
    lines = [("".join(CJK.findall(l))) for l in text.splitlines()]
    lines = [l for l in lines if l]
    # a) n lines of exactly `per`
    cand = [l for l in lines if len(l) == per]
    if len(cand) >= n:
        return "".join(cand[:n]), "EXTRACTED"
    # b) n/2 lines of 2*per (couplet per line)
    cand = [l for l in lines if len(l) == 2 * per]
    if len(cand) >= n // 2:
        return "".join(cand[: n // 2]), "EXTRACTED"
    all_cjk = "".join(lines)
    if not all_cjk:
        return "", "NO_POEM"
    if len(all_cjk) == per * n:
        return all_cjk, "EXTRACTED"
    return all_cjk, "WRONG_LENGTH"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter", help="LoRA adapter directory (mechanism E). Absent = the baseline.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    import mlx.core as mx
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    from verify_form import Rime, verify

    mx.random.seed(a.seed)
    rime = Rime(os.path.join(HERE, "rime", "pingshui.json"))
    t0 = time.time()
    kw = {"adapter_path": a.adapter} if a.adapter else {}
    model, tok = load(a.model, **kw)
    print(f"loaded {a.model} adapter={a.adapter or 'NONE'} in {time.time()-t0:.1f}s", flush=True)
    sampler = make_sampler(temp=a.temp)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    rows = []
    with open(a.out, "w") as fh:
        for form, per, n in FORMS:
            for topic in TOPICS:
                p = PROMPT.format(topic=topic, form=form, n=n, per=per)
                msgs = [{"role": "user", "content": p}]
                text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
                t = time.time()
                out = generate(model, tok, prompt=text, max_tokens=a.max_tokens,
                               sampler=sampler, verbose=False)
                poem, bucket = extract(out, per, n)
                v = verify(poem, rime) if bucket == "EXTRACTED" else None
                # The extractor strips to CJK, so a Latin token wedged inside a line is INVISIBLE
                # to the verifier by construction — "不 SwiftUI 也在斜" is 28 CJK characters and
                # scores as a 七絕. It is counted here instead of being left for someone to find.
                r = {"topic": topic, "form": form, "adapter": a.adapter, "prompt": p, "raw": out,
                     "latin_intrusion": bool(LATIN.search(out)),
                     "poem": poem, "bucket": bucket,
                     "pass": bool(v and v["pass"]), "verify": v,
                     "secs": round(time.time() - t, 2)}
                rows.append(r)
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                fh.flush()
            done = [r for r in rows if r["form"] == form]
            ok = sum(1 for r in done if r["pass"])
            print(f"  {form}: {ok}/{len(done)} pass · "
                  f"extracted {sum(1 for r in done if r['bucket']=='EXTRACTED')} · "
                  f"wrong_length {sum(1 for r in done if r['bucket']=='WRONG_LENGTH')} · "
                  f"no_poem {sum(1 for r in done if r['bucket']=='NO_POEM')}", flush=True)

    rep = {"model": a.model, "adapter": a.adapter, "temp": a.temp, "seed": a.seed,
           "n_prompts": len(rows),
           "by_form": {}}
    for form, per, n in FORMS:
        d = [r for r in rows if r["form"] == form]
        ex = [r for r in d if r["bucket"] == "EXTRACTED"]
        rep["by_form"][form] = {
            "n": len(d), "extracted": len(ex),
            "wrong_length": sum(1 for r in d if r["bucket"] == "WRONG_LENGTH"),
            "no_poem": sum(1 for r in d if r["bucket"] == "NO_POEM"),
            "pass_over_all_attempts": sum(1 for r in d if r["pass"]) / len(d),
            "pass_over_extracted": (sum(1 for r in ex if r["pass"]) / len(ex)) if ex else None,
            "latin_intrusion": sum(1 for r in d if r["latin_intrusion"]),
            "latin_intrusion_that_still_PASSED": sum(1 for r in d if r["latin_intrusion"] and r["pass"]),
            "failed_by_rule": {},
        }
        for r in ex:
            for k, rl in r["verify"]["rules"].items():
                if not rl["pass"]:
                    rep["by_form"][form]["failed_by_rule"][k] = \
                        rep["by_form"][form]["failed_by_rule"].get(k, 0) + 1
    json.dump(rep, open(a.report, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(rep["by_form"], ensure_ascii=False, indent=1))
    print("->", a.out, a.report)


if __name__ == "__main__":
    main()
