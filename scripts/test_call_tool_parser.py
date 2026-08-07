#!/usr/bin/env python
"""Test parse_call_tool_xml against DR-Tulu's mentor prompt syntax.

Covers every literal example in paper_notes/system_prompts/drtulu_prompt.txt,
plus realistic model-output variations and adversarial cases. A silent parse
failure here costs an entire 830-question run (0% SR, no error anywhere), so
the bar is: every documented form MUST parse, and malformed input MUST return
None rather than a wrong tool call.
"""
import sys

sys.path.insert(0, "/work/pi_skrastanov_umass_edu/ritesh/graph_research/repos/OpenResearcher")
from deploy_agent import parse_call_tool_xml  # noqa: E402

FAILS = []


def check(desc, got, want):
    ok = got == want
    print(f"  [{'PASS' if ok else 'FAIL'}] {desc}")
    if not ok:
        print(f"         got:  {got}")
        print(f"         want: {want}")
        FAILS.append(desc)


print("=" * 70)
print("A. Literal examples from the mentor's DR-Tulu prompt")
print("=" * 70)
check("search with topn",
      parse_call_tool_xml('<call_tool name="search" topn="10">your search query</call_tool>'),
      {"name": "search", "arguments": {"topn": 10, "query": "your search query"}})

check("open by cursor+id, empty body",
      parse_call_tool_xml('<call_tool name="open" cursor="0" id="3"></call_tool>'),
      {"name": "open", "arguments": {"cursor": 0, "id": "3"}})

check("open with loc+num_lines",
      parse_call_tool_xml('<call_tool name="open" cursor="1" loc="40" num_lines="20"></call_tool>'),
      {"name": "open", "arguments": {"cursor": 1, "loc": 40, "num_lines": 20}})

check("open a URL via body",
      parse_call_tool_xml('<call_tool name="open">https://example.com/document</call_tool>'),
      {"name": "open", "arguments": {"id": "https://example.com/document"}})

check("find with cursor",
      parse_call_tool_xml('<call_tool name="find" cursor="1">exact text pattern</call_tool>'),
      {"name": "find", "arguments": {"cursor": 1, "pattern": "exact text pattern"}})

print()
print("=" * 70)
print("B. Realistic model output (thinking, prose, whitespace, newlines)")
print("=" * 70)
check("preceded by <think> block",
      parse_call_tool_xml(
          '<think>I should search the corpus first.</think>\n'
          '<call_tool name="search" topn="5">Queen Arwa University</call_tool>'),
      {"name": "search", "arguments": {"topn": 5, "query": "Queen Arwa University"}})

check("multiline body is stripped",
      parse_call_tool_xml('<call_tool name="search">\n  multi line query\n</call_tool>'),
      {"name": "search", "arguments": {"query": "multi line query"}})

check("single-quoted attributes",
      parse_call_tool_xml("<call_tool name='search' topn='7'>q</call_tool>"),
      {"name": "search", "arguments": {"topn": 7, "query": "q"}})

check("unquoted attribute values",
      parse_call_tool_xml('<call_tool name=find cursor=2>needle</call_tool>'),
      {"name": "find", "arguments": {"cursor": 2, "pattern": "needle"}})

check("extra whitespace inside the tag",
      parse_call_tool_xml('<call_tool   name="search"   topn="3" >  spaced  </call_tool>'),
      {"name": "search", "arguments": {"topn": 3, "query": "spaced"}})

check("trailing prose after the call is ignored",
      parse_call_tool_xml('<call_tool name="search">abc</call_tool>\nNow let me look at that.'),
      {"name": "search", "arguments": {"query": "abc"}})

check("only the FIRST call is taken (scaffold executes one per turn)",
      parse_call_tool_xml('<call_tool name="search">first</call_tool>'
                          '<call_tool name="search">second</call_tool>'),
      {"name": "search", "arguments": {"query": "first"}})

check("query containing angle brackets / quotes",
      parse_call_tool_xml('<call_tool name="search">a > b "quoted"</call_tool>'),
      {"name": "search", "arguments": {"query": 'a > b "quoted"'}})

print()
print("=" * 70)
print("C. Adversarial / malformed -- MUST return None, never a wrong call")
print("=" * 70)
check("no call_tool at all", parse_call_tool_xml("Just some reasoning text."), None)
check("empty string", parse_call_tool_xml(""), None)
check("None input", parse_call_tool_xml(None), None)
check("unknown tool name",
      parse_call_tool_xml('<call_tool name="delete_everything">x</call_tool>'), None)
check("missing name attribute",
      parse_call_tool_xml('<call_tool topn="5">query</call_tool>'), None)
check("unclosed tag", parse_call_tool_xml('<call_tool name="search">query'), None)
check("search with no query (body empty, no attr)",
      parse_call_tool_xml('<call_tool name="search"></call_tool>'), None)
check("find with no pattern",
      parse_call_tool_xml('<call_tool name="find" cursor="1"></call_tool>'), None)
check("non-numeric topn is dropped, not crashed",
      parse_call_tool_xml('<call_tool name="search" topn="many">q</call_tool>'),
      {"name": "search", "arguments": {"query": "q"}})
check("attribute not valid for this tool is dropped",
      parse_call_tool_xml('<call_tool name="find" topn="9" cursor="1">p</call_tool>'),
      {"name": "find", "arguments": {"cursor": 1, "pattern": "p"}})
check("open with no args at all is still valid (re-display page)",
      parse_call_tool_xml('<call_tool name="open"></call_tool>'),
      {"name": "open", "arguments": {}})
check("explicit id attr wins over body",
      parse_call_tool_xml('<call_tool name="open" id="7">https://ignored.example</call_tool>'),
      {"name": "open", "arguments": {"id": "7"}})
check("case-insensitive tag and name",
      parse_call_tool_xml('<CALL_TOOL NAME="SEARCH">Q</CALL_TOOL>'),
      {"name": "search", "arguments": {"query": "Q"}})

print()
print("=" * 70)
print("D. Must NOT hijack DR-Venus / LiteResearcher JSON format")
print("=" * 70)
check("JSON tool_call is left alone (returns None -> falls through)",
      parse_call_tool_xml('<tool_call>{"name": "search", "arguments": {"query": "x"}}</tool_call>'),
      None)

print()
print("=" * 70)
if FAILS:
    print(f"RESULT: {len(FAILS)} FAILURE(S)")
    for f in FAILS:
        print(f"  - {f}")
    sys.exit(1)
print("RESULT: ALL PASSED — parser handles every documented form,")
print("        rejects malformed input, and does not touch the JSON path.")
sys.exit(0)
