#!/usr/bin/env python
"""Health validation for a BCP run's trajectories.

Distinguishes "the model performed badly" (a valid result) from "the harness
broke" (an invalid result that must not reach Table 1). Low SR alone is NOT a
failure signal -- BrowseComp-Plus is hard. These checks look for the signatures
of a BROKEN run instead.

Usage:  python validate_run.py results/<run_dir> [--system drtulu|drvenus|literes]
Exit 0 = healthy, 1 = suspicious (investigate before trusting numbers).
"""
import argparse
import collections
import glob
import json
import os
import re
import sys

# A run is suspicious if it trips any of these.
THRESHOLDS = {
    "min_tool_call_rate": 0.80,     # frac of questions making >=1 tool call
    "max_zero_tool_frac": 0.20,     # frac of questions with NO tool calls at all
    "max_parse_fail_frac": 0.20,    # frac of assistant turns with no parsed call
    "max_empty_turn_frac": 0.15,    # frac of assistant turns that are empty
    "max_capped_frac": 0.90,        # frac hitting max_rounds (all capped = stuck)
    "min_answer_tag_frac": 0.10,    # frac producing a real <answer> tag
    "max_error_frac": 0.05,         # frac with status=fail
    "max_nonlatin_frac": 0.25,      # frac showing heavy language drift
    "min_unique_query_frac": 0.50,  # query diversity (low = looping)
}


def load(run_dir):
    rows = []
    for f in glob.glob(os.path.join(run_dir, "node_*_shard_*.jsonl")):
        for line in open(f, encoding="utf-8"):
            try:
                rows.append(json.loads(line))
            except Exception:
                pass          # torn line from a preemption; resume will redo it
    return rows


def analyse(rows):
    n = len(rows)
    if not n:
        return None

    stats = collections.Counter()
    all_turns = parse_fail = empty_turns = 0
    queries, tools_used, reasons = [], collections.Counter(), collections.Counter()
    nonlatin_qs = 0

    for r in rows:
        msgs = r.get("messages", [])
        reasons[r.get("terminated_reason", "?")] += 1
        if r.get("status") == "fail":
            stats["error"] += 1

        asst = [m for m in msgs if m.get("role") == "assistant"]
        all_turns += len(asst)
        ncalls = 0
        blob = ""
        for m in asst:
            tcs = m.get("tool_calls") or []
            content = (m.get("content") or "")
            blob += content + json.dumps(tcs, ensure_ascii=False)
            if tcs:
                ncalls += len(tcs)
                for tc in tcs:
                    fn = tc["function"]["name"].split(".")[-1]
                    tools_used[fn] += 1
                    a = tc["function"]["arguments"]
                    if isinstance(a, str):
                        try:
                            a = json.loads(a)
                        except Exception:
                            a = {}
                    if fn == "search" and a.get("query"):
                        queries.append(str(a["query"]))
            else:
                if not content.strip():
                    empty_turns += 1
                else:
                    parse_fail += 1

        if ncalls == 0:
            stats["zero_tool"] += 1
        if len(re.findall(r"[一-鿿؀-ۿ֐-׿]", blob)) > 50:
            nonlatin_qs += 1

        last = ""
        for m in reversed(msgs):
            if m.get("role") == "assistant" and (m.get("content") or "").strip():
                last = m["content"]
                break
        if "<answer>" in last and "</answer>" in last:
            stats["answer_tag"] += 1

    return {
        "n": n,
        "zero_tool_frac": stats["zero_tool"] / n,
        "tool_call_rate": 1 - stats["zero_tool"] / n,
        "parse_fail_frac": parse_fail / max(1, all_turns),
        "empty_turn_frac": empty_turns / max(1, all_turns),
        "capped_frac": reasons.get("max_rounds", 0) / n,
        "answer_tag_frac": stats["answer_tag"] / n,
        "error_frac": stats["error"] / n,
        "nonlatin_frac": nonlatin_qs / n,
        "unique_query_frac": (len(set(queries)) / len(queries)) if queries else 1.0,
        "mean_turns": all_turns / n,
        "tools_used": dict(tools_used),
        "terminated_reasons": dict(reasons),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--system", default="")
    args = ap.parse_args()

    rows = load(args.run_dir)
    s = analyse(rows)
    if not s:
        print(f"NO RECORDS in {args.run_dir}")
        sys.exit(1)

    print("=" * 66)
    print(f"RUN VALIDATION  {args.run_dir}   ({s['n']} questions)")
    print("=" * 66)
    print(f"  mean turns/question      {s['mean_turns']:.1f}")
    print(f"  tools used               {s['tools_used']}")
    print(f"  terminated_reasons       {s['terminated_reasons']}")
    print()

    problems = []
    checks = [
        ("questions making tool calls", s["tool_call_rate"], ">=", THRESHOLDS["min_tool_call_rate"]),
        ("questions with ZERO tools",   s["zero_tool_frac"], "<=", THRESHOLDS["max_zero_tool_frac"]),
        ("turns failing to parse",      s["parse_fail_frac"], "<=", THRESHOLDS["max_parse_fail_frac"]),
        ("empty assistant turns",       s["empty_turn_frac"], "<=", THRESHOLDS["max_empty_turn_frac"]),
        ("hitting max_rounds",          s["capped_frac"], "<=", THRESHOLDS["max_capped_frac"]),
        ("producing <answer> tag",      s["answer_tag_frac"], ">=", THRESHOLDS["min_answer_tag_frac"]),
        ("status=fail",                 s["error_frac"], "<=", THRESHOLDS["max_error_frac"]),
        ("heavy language drift",        s["nonlatin_frac"], "<=", THRESHOLDS["max_nonlatin_frac"]),
        ("unique search queries",       s["unique_query_frac"], ">=", THRESHOLDS["min_unique_query_frac"]),
    ]
    for name, val, op, thr in checks:
        good = val >= thr if op == ">=" else val <= thr
        if not good:
            problems.append(f"{name}: {val:.1%} (want {op} {thr:.0%})")
        print(f"  [{'OK  ' if good else 'WARN'}] {name:<28} {val:>6.1%}  (want {op} {thr:.0%})")

    # --- checks that trajectory STRUCTURE alone cannot catch -----------------
    # (added after this validator passed a run known to be broken)

    # 1. terminated_reason must be populated. If it is '?', the records predate
    #    the patch or the field is not being written -- we cannot tell a capped
    #    episode from a clean answer, so the run is not analysable.
    if s["terminated_reasons"].get("?", 0):
        problems.append(
            f"terminated_reason missing on {s['terminated_reasons']['?']} records "
            "-- run predates the schema patch or field is not being written")

    # 2. Degenerate final answers: mixed-script gibberish in the ANSWER itself
    #    is a stronger signal than drift anywhere in the trajectory.
    bad_ans = 0
    for r in rows:
        last = ""
        for m in reversed(r.get("messages", [])):
            if m.get("role") == "assistant" and (m.get("content") or "").strip():
                last = m["content"]
                break
        ans = re.search(r"<answer>(.*?)</answer>", last, re.DOTALL)
        if ans and len(re.findall(r"[一-鿿؀-ۿ֐-׿]", ans.group(1))) > 5:
            bad_ans += 1
    frac_bad = bad_ans / s["n"]
    good = frac_bad <= 0.10
    if not good:
        problems.append(f"degenerate final answers (mixed-script): {frac_bad:.1%}")
    print(f"  [{'OK  ' if good else 'WARN'}] {'degenerate final answers':<28} "
          f"{frac_bad:>6.1%}  (want <= 10%)")

    # 2b. Malformation accounting. A rising DROP rate means the salvage path has
    #     stopped keeping up with how this model malforms its output -- worth
    #     knowing before it costs a whole run. The repaired portion is expected
    #     and is a reported finding, not a fault.
    att = sum(r.get("tool_calls_attempted", 0) for r in rows)
    rep = sum(r.get("tool_calls_repaired", 0) for r in rows)
    drp = sum(r.get("tool_calls_dropped", 0) for r in rows)
    if att:
        mal_rate = (rep + drp) / att
        drop_rate = drp / att
        print(f"  [INFO] tool calls: {att} attempted, {rep} repaired, {drp} dropped")
        print(f"  [INFO] {'malformation rate':<28} {mal_rate:>6.1%}  (reported, not a fault)")
        good = drop_rate <= 0.25
        if not good:
            problems.append(f"unrecoverable tool-call drops: {drop_rate:.1%} "
                            "(salvage may need extending for this model)")
        print(f"  [{'OK  ' if good else 'WARN'}] {'unrecoverable drop rate':<28} "
              f"{drop_rate:>6.1%}  (want <= 25%)")
    else:
        print("  [SKIP] malformation accounting     (no tool_calls_attempted field)")

    # 3. HTTP 400s are invisible in trajectories -- check the vLLM log directly.
    vlog = os.environ.get("VLLM_LOG", "")
    if vlog and os.path.exists(vlog):
        txt = open(vlog, errors="ignore").read()
        n400 = txt.count('"POST /v1/completions HTTP/1.1" 400')
        n200 = txt.count('"POST /v1/completions HTTP/1.1" 200')
        frac = n400 / max(1, n400 + n200)
        good = frac <= 0.005
        if not good:
            problems.append(f"HTTP 400s from vLLM: {n400}/{n400+n200} ({frac:.1%}) "
                            "-- max_tokens clamp may be failing")
        print(f"  [{'OK  ' if good else 'WARN'}] {'vLLM HTTP-400 rate':<28} "
              f"{frac:>6.1%}  (want <= 0.5%)  [{n400}/{n400+n200}]")
    else:
        print("  [SKIP] vLLM HTTP-400 rate            (set VLLM_LOG to enable)")

    # System-specific: the tool the prompt teaches must actually be getting used.
    if args.system == "drtulu" and not s["tools_used"]:
        problems.append("DR-Tulu: NO tool calls parsed at all -- "
                        "<call_tool> XML parser is probably missing")

    print()
    if problems:
        print("VERDICT: SUSPICIOUS -- investigate before trusting these numbers")
        for p in problems:
            print(f"    - {p}")
        print("\n  NOTE: low SR by itself is NOT a bug. These flags indicate the")
        print("        HARNESS may be broken, which is a different problem.")
        sys.exit(1)
    print("VERDICT: HEALTHY -- agent is exercising the tools correctly.")
    print("  Whatever SR this run produces is a valid measurement.")
    sys.exit(0)


if __name__ == "__main__":
    main()
