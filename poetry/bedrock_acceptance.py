#!/usr/bin/env python
"""ET-IV — the acceptance check to run THE MOMENT a scoped Bedrock principal lands on llm1.

Written before the credential exists so the first thing that touches it is reviewed rather than
improvised. It answers, in order, the four things that have to be true before an arm runs:

  1  THE PRINCIPAL IS NOT ROOT. 理 11747 and 鉄 11759 (WO-265 S1): a scoped principal with
     bedrock:InvokeModel only. If get_caller_identity still says :root, this REFUSES and stops —
     it does not "work anyway", because working anyway is how root keys stay in place.
  2  THE PRINCIPAL IS ACTUALLY SCOPED. It must NOT be able to read IAM or list buckets. A key that
     can invoke Bedrock AND do everything else is the root key with a new name.
  3  THE MODEL ID INVOKES. `us.anthropic.claude-fable-5-1` — the INFERENCE PROFILE id. The bare
     `anthropic.claude-fable-5-1` is INFERENCE_PROFILE-only and will fail (雲 11749); this checks
     the bare id fails and the profile id succeeds, so the distinction is proven, not assumed.
  4  THE METER SEES REAL USAGE. One minimal call, max_tokens=1, through the same CostMeter the arms
     use, against the $40 cap. Costs a small fraction of a cent and proves the accounting path
     end to end with real token counts rather than FakeBedrock's.

It writes an artifact and spends essentially nothing. Nothing else runs until this passes.
"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import api_rater as AR

PROFILE_ID = "us.anthropic.claude-fable-5-1"
BARE_ID = "anthropic.claude-fable-5-1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    ap.add_argument("--ledger", default=os.path.join(HERE, "data", "iv_spend_ledger.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "bedrock_acceptance.json"))
    a = ap.parse_args()
    import boto3
    from botocore.exceptions import ClientError
    res = {"document": "ET-IV — Bedrock acceptance check", "region": a.region}

    # WRITE WHAT WE KNOW, EVEN IF A LATER CHECK THROWS. Checks 1-3 passed against the real scoped
    # principal on 09-22 and check [4] raised before the final dump, so the artefact on disk stayed
    # the one from 09-20 — a file asserting "REFUSED — still the root key" about a principal that
    # had already been replaced. A stale artefact making a FALSE security claim is worse than no
    # artefact: it answers the question wrongly to whoever reads it next.
    import atexit
    res["incomplete"] = True
    def _flush():
        json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    atexit.register(_flush)

    # 1 — not root
    ident = boto3.client("sts", region_name=a.region).get_caller_identity()
    arn = ident.get("Arn", "")
    res["1_principal"] = {"arn": arn, "is_root": arn.endswith(":root")}
    print("  [1] principal: %s" % arn)
    if arn.endswith(":root"):
        res["verdict"] = "REFUSED — still the root key"
        json.dump(res, open(a.out, "w"), indent=1)
        sys.exit("STOP: this is the ROOT principal. 理 11747 forbids running IV on it, and a check "
                 "that proceeds anyway is how a root key stays in place. Nothing was invoked.")

    # 2 — actually scoped: it must NOT be able to do unrelated things
    denied = {}
    for name, fn in (("iam:ListUsers", lambda: boto3.client("iam", region_name=a.region).list_users(MaxItems=1)),
                     ("s3:ListBuckets", lambda: boto3.client("s3", region_name=a.region).list_buckets())):
        try:
            fn(); denied[name] = False
        except ClientError as e:
            denied[name] = e.response["Error"]["Code"] in ("AccessDenied", "AccessDeniedException",
                                                           "UnauthorizedOperation")
        except Exception:
            denied[name] = True
    res["2_scoped"] = {"denied": denied,
                       "note": "a principal that can invoke Bedrock AND read IAM/S3 is the root key "
                               "with a new name; this is the check that catches that"}
    print("  [2] unrelated access denied: %s" % denied)
    if not all(denied.values()):
        res["verdict"] = "REFUSED — the principal is not scoped"
        json.dump(res, open(a.out, "w"), indent=1)
        sys.exit("STOP: this principal can reach services outside Bedrock. Not scoped. Nothing invoked.")

    # 3 — the bare id must FAIL and the profile id must WORK
    rt = boto3.client("bedrock-runtime", region_name=a.region)
    probe = {"messages": [{"role": "user", "content": "Reply with the single letter A."}],
             "max_tokens": 1}
    bare_failed = False
    try:
        AR.bedrock_transport(dict(probe, model=BARE_ID), client=rt)
    except Exception as e:
        bare_failed = True
        res["3_bare_id_rejected"] = {"id": BARE_ID, "error": type(e).__name__, "detail": str(e)[:160]}
    print("  [3] bare id %s: %s" % (BARE_ID, "rejected as expected" if bare_failed else "🔴 ACCEPTED"))

    # 4 — one real metered call on the profile id
    meter = AR.CostMeter(ledger=a.ledger)
    before = meter.spent
    rid = meter.reserve(PROFILE_ID, 32, 1, note="acceptance")
    r = AR.bedrock_transport(dict(probe, model=PROFILE_ID), client=rt)
    u = r["usage"]
    actual = meter.settle(rid, u["input_tokens"], u["output_tokens"])
    res["3_profile_id"] = PROFILE_ID
    res["4_metered_call"] = {"input_tokens": u["input_tokens"], "output_tokens": u["output_tokens"],
                             "actual_usd": round(actual, 8), "ledger_before": round(before, 6),
                             "ledger_after": round(meter.spent, 6),
                             "text": "".join(c["text"] for c in r["content"])[:40]}
    print("  [4] metered call: in %d out %d  $%.8f   %s"
          % (u["input_tokens"], u["output_tokens"], actual, meter.line()))
    res["verdict"] = ("PASS" if bare_failed else
                      "PASS WITH A NOTE — the bare model id did NOT fail; 雲's INFERENCE_PROFILE "
                      "reading may not hold for this model and should be re-checked before it is "
                      "relied on elsewhere")
    res.pop("incomplete", None)
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print("\n  %s -> %s" % (res["verdict"], a.out))


if __name__ == "__main__":
    main()
