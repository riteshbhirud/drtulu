#!/usr/bin/env python
"""Assemble Table 1 (BrowseComp-Plus / BM25) from judged runs.

Reads results/<run>/judge_summary.json for every completed run and emits both a
markdown table and a JSON blob. Runs that have not been judged yet show as "--",
so this is safe to run at any point to see current progress.

SR    = judge_correct / n_judged * 100
Turns = mean rounds_used   (one round = one LLM call, per the run spec)

Usage: python build_table1.py [--out results/table1_umass_bm25.md]
"""
import argparse
import glob
import json
import os

ROOT = "/work/pi_skrastanov_umass_edu/ritesh/graph_research"

# Mentor's batch order; (label, run-dir prefix)
LAYOUT = [
    ("LiteResearcher-4B", "literes"),
    ("DR-Tulu-8B", "drtulu"),
    ("DR-Venus-4B", "venus"),
]
STAGES = [("Base", "base"), ("SFT", "sft"), ("RL", "rl")]


def load(run):
    p = os.path.join(ROOT, "results", run, "judge_summary.json")
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


def raw_progress(run):
    """Inference progress even when not yet judged."""
    n = 0
    for f in glob.glob(os.path.join(ROOT, "results", run, "node_*_shard_*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            n += sum(1 for _ in fh)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results/table1_umass_bm25.md"))
    args = ap.parse_args()

    rows, blob = [], {}
    for label, pref in LAYOUT:
        for stage_label, stage in STAGES:
            run = f"{pref}_{stage}_bm25_top5"
            s = load(run)
            done = raw_progress(run)
            if s:
                sr = f"{s['success_rate_pct']:.1f}%"
                tn = f"{s['avg_turns']:.1f}"
                note = "" if s["n_judged"] == 830 else f" (n={s['n_judged']})"
                blob[run] = s
            else:
                sr = tn = "--"
                note = f" ({done}/830 run, unjudged)" if done else ""
            rows.append((label, stage_label, sr, tn, note))

    lines = [
        "# Table 1 — BrowseComp-Plus (BM25, top-5), UMass systems",
        "",
        "SR = success rate (official BC-Plus judge, Qwen3-32B).",
        "Turns = mean LLM calls per question (max_rounds=100).",
        "",
        "**No published BrowseComp-Plus baseline exists for any of these three",
        "systems. Every number here is a new measurement.**",
        "",
        "Oracle ceiling for BM25 top-5 on this benchmark: **70.7%** "
        "(gold doc retrievable when querying with the answer string); "
        "gold at rank 1 for 56.7%. SR should be read against that ceiling.",
        "",
        "| System | Stage | SR ↑ | Turns ↓ | |",
        "|---|---|---:|---:|---|",
    ]
    prev = None
    for label, stage, sr, tn, note in rows:
        shown = label if label != prev else ""
        prev = label
        lines.append(f"| {shown} | {stage} | {sr} | {tn} |{note} |")
    lines += [
        "",
        "## Method notes",
        "",
        "- Scaffold: TIGER-AI-Lab/OpenResearcher, mentor's adapted "
        "`search`/`open`/`find` prompts (uniform tool interface across systems).",
        "- Retriever: prebuilt Tevatron BM25 Lucene index, top-5, "
        "512-token-equivalent snippets, pyserini defaults (k1=0.9, b=0.4).",
        "- Judge: Qwen3-32B with the official BC-Plus grader prompt; "
        "unparseable judgements count as incorrect.",
        "- Tool-call malformation is repaired where intent is unambiguous and "
        "reported per trajectory; see paper_notes/malformation.md.",
        "- All scaffold modifications are documented in "
        "paper_notes/scaffold_patches.md with measured impact.",
    ]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(args.out.replace(".md", ".json"), "w") as f:
        json.dump({"runs": blob, "note_no_published_bcp_baseline": True,
                   "bm25_oracle_ceiling_top5_pct": 70.7,
                   "bm25_oracle_rank1_pct": 56.7}, f, indent=2)

    print("\n".join(lines))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
