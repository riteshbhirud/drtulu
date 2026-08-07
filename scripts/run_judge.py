#!/usr/bin/env python
"""Judge BCP trajectories with Qwen3-32B using the OFFICIAL BrowseComp-Plus prompt.

Run as a SEPARATE pass after inference (Step 13): one A100 cannot hold both the
agent's model and a 32B judge.

Answer extraction follows Step 12, per system:
  LiteResearcher : <answer>...</answer>, do NOT unwrap \\boxed{}
  DR-Tulu        : <answer>...</answer>, strip <cite id=...> tags, unwrap \\boxed{}
  DR-Venus       : <answer>...</answer>

Both raw_final_output and extracted_final_answer are saved, so grading is
auditable and can be recomputed without re-running inference.

Usage:
  python run_judge.py --run_dir results/venus_rl_bm25_top5 [--limit N]
"""
import argparse
import glob
import json
import os
import re
import sys

# Verbatim from repos/BrowseComp-Plus/scripts_evaluation/evaluate_run.py
GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response].

[correct_answer]: Repeat the [correct_answer] given above.

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], in the context of this [question]. You should judge whether the extracted_final_answer is semantically equivalent to [correct_answer], allowing the extracted_final_answer to be string variations of [correct_answer]. You should also allow the extracted_final_answer to be more precise or verbose than [correct_answer], as long as its additional details are correct. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers are semantically equivalent.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\\%| and 100|\\%| from [response]. Put 100 if there is no confidence score available.
""".strip()

_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_CITE = re.compile(r"</?cite[^>]*>", re.IGNORECASE)
_BOXED = re.compile(r"\\boxed\{(.*?)\}", re.DOTALL)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def raw_final_output(rec):
    """Last assistant turn carrying real content."""
    for m in reversed(rec.get("messages", [])):
        if m.get("role") == "assistant" and (m.get("content") or "").strip():
            return m["content"]
    return ""


def extract_answer(raw, system):
    """Step 12 extraction, per system. Returns "" when no answer was produced."""
    if not raw:
        return ""
    text = _THINK.sub("", raw)
    m = _ANSWER.findall(text)
    ans = m[-1].strip() if m else ""      # last <answer> wins
    if not ans:
        return ""
    s = (system or "").lower()
    if "tulu" in s:
        # strip citation markup, then unwrap \boxed{} if present
        ans = _CITE.sub("", ans).strip()
        b = _BOXED.search(ans)
        if b:
            ans = b.group(1).strip()
    # LiteResearcher: explicitly do NOT unwrap \boxed{} (spec Step 12)
    # DR-Venus: plain <answer> content
    return ans.strip()


def parse_judgement(text):
    """Official BCP parsing: cascading regex, bold-markdown tolerant."""
    out = {"extracted_final_answer": None, "reasoning": None,
           "correct": None, "confidence": None, "parse_error": False}
    for pat in (r"\*\*correct:\*\*\s*(yes|no)", r"\*\*correct\*\*:\s*(yes|no)",
                r"correct:\s*(yes|no)"):
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            out["correct"] = m.group(1).lower() == "yes"
            break
    for pat in (r"\*\*extracted_final_answer:\*\*\s*(.*?)(?=\n|$)",
                r"\*\*extracted_final_answer\*\*:\s*(.*?)(?=\n|$)",
                r"extracted_final_answer:\s*(.*?)(?=\n|$)"):
        m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if m:
            out["extracted_final_answer"] = m.group(1).strip()
            break
    m = re.search(r"reasoning:\s*(.*?)(?=\ncorrect:|\Z)", text,
                  re.IGNORECASE | re.DOTALL)
    if m:
        out["reasoning"] = m.group(1).strip()[:2000]
    for pat in (r"\*\*confidence:\*\*\s*(\d+(?:\.\d+)?)", r"confidence:\s*(\d+(?:\.\d+)?)"):
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            out["confidence"] = min(100.0, float(m.group(1)))
            break
    out["parse_error"] = out["correct"] is None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--judge_model", default="models/Qwen3-32B")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_model_len", type=int, default=32768)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    args = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(os.path.join(args.run_dir, "node_*_shard_*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print(f"No records in {args.run_dir}")
        sys.exit(1)
    print(f"loaded {len(rows)} trajectories from {args.run_dir}")

    system = rows[0].get("system", "")
    prompts, meta = [], []
    for r in rows:
        raw = raw_final_output(r)
        ext = extract_answer(raw, system or args.run_dir)
        # Judge the EXTRACTED answer when we have one; otherwise the raw turn,
        # so a malformed-but-present answer still gets a fair reading.
        response = ext if ext else raw
        prompts.append(GRADER_TEMPLATE.format(
            question=r.get("question", ""), response=response,
            correct_answer=r.get("answer", "")))
        meta.append((r, raw, ext))

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.judge_model, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization,
              tensor_parallel_size=args.tensor_parallel_size,
              trust_remote_code=True)
    # Official BCP judge decoding, thinking disabled.
    sp = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, max_tokens=4096)
    convs = [[{"role": "user", "content": p}] for p in prompts]
    outs = llm.chat(convs, sp, chat_template_kwargs={"enable_thinking": False})

    judged, n_correct, n_parse_err = [], 0, 0
    for (r, raw, ext), o in zip(meta, outs):
        txt = o.outputs[0].text
        j = parse_judgement(txt)
        n_correct += 1 if j["correct"] else 0
        n_parse_err += 1 if j["parse_error"] else 0
        rec = dict(r)
        rec.update({
            "raw_final_output": raw,
            "extracted_final_answer": ext,
            "judge_correct": bool(j["correct"]),
            "judge_reasoning": j["reasoning"],
            "judge_confidence": j["confidence"],
            "judge_extracted_answer": j["extracted_final_answer"],
            "judge_parse_error": j["parse_error"],
            "judge_model": args.judge_model,
            "judge_prompt": "browsecomp-plus-official",
        })
        rec.pop("messages", None)          # keep the judged file small
        judged.append(rec)

    out_path = os.path.join(args.run_dir, "judged.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in judged:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # SR is over ALL judged records; unparseable judgements count as incorrect,
    # matching the official harness.
    sr = n_correct / len(judged) * 100
    turns = sum(r.get("rounds_used", 0) for r in judged) / len(judged)
    summary = {
        "run_dir": args.run_dir,
        "system": system,
        "checkpoint": rows[0].get("checkpoint", ""),
        "retriever": rows[0].get("retriever", "bm25"),
        "dataset": "browsecomp-plus",
        "n_judged": len(judged),
        "n_correct": n_correct,
        "success_rate_pct": round(sr, 2),
        "avg_turns": round(turns, 2),
        "judge_model": args.judge_model,
        "judge_parse_errors": n_parse_err,
        "note_no_published_bcp_baseline": True,
    }
    with open(os.path.join(args.run_dir, "judge_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 56)
    for k, v in summary.items():
        print(f"  {k:32s} {v}")
    print("=" * 56)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
