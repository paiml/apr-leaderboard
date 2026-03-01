#!/usr/bin/env python3
"""Extract instruction/response pairs from ground truth Python corpora.

Reads Python files from the four ground truth corpora (depyler, hf-gtc,
jax-gtc, vllm-gtc), extracts function/class definitions with docstrings,
and writes JSONL suitable for `apr finetune --task instruct`.

Output format:
    {"instruction": "...", "response": "...", "metadata": {"source": "...", "file": "..."}}

Usage:
    python3 scripts/prep-instruct-data.py [--output data/instruct-corpus.jsonl]
    python3 scripts/prep-instruct-data.py --corpus depyler --output data/depyler.jsonl
    python3 scripts/prep-instruct-data.py --min-response-lines 3 --max-response-lines 200
"""

import argparse
import ast
import json
import os
import sys
import textwrap
from pathlib import Path

# Default corpus roots relative to this repo's parent
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_ROOT = REPO_ROOT.parent  # ~/src

CORPORA = {
    "depyler": [
        SRC_ROOT / "depyler" / "examples",
        SRC_ROOT / "depyler" / "tdd-book" / "tests",
    ],
    "hf-gtc": [
        SRC_ROOT / "hf-ground-truth-corpus" / "src",
    ],
    "jax-gtc": [
        SRC_ROOT / "jax-ground-truth-corpus" / "src",
    ],
    "vllm-gtc": [
        SRC_ROOT / "vllm-ground-truth-corpus" / "src",
    ],
}


def extract_functions_from_file(filepath: Path, source: str) -> list[dict]:
    """Extract function/class definitions with docstrings from a Python file.

    Returns list of {"instruction": ..., "response": ..., "metadata": ...} dicts.
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except (OSError, UnicodeDecodeError):
        return []

    try:
        tree = ast.parse(content, filename=str(filepath))
    except SyntaxError:
        return []

    lines = content.splitlines(keepends=True)
    samples = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue

        docstring = ast.get_docstring(node)
        if not docstring or len(docstring.strip()) < 10:
            continue

        # Extract the full source of this node
        start = node.lineno - 1
        end = node.end_lineno if hasattr(node, "end_lineno") and node.end_lineno else start + 1
        node_source = "".join(lines[start:end])
        node_source = textwrap.dedent(node_source).strip()

        if not node_source:
            continue

        # Build instruction from the signature + docstring
        if isinstance(node, ast.ClassDef):
            kind = "class"
            instruction = f"Write a Python {kind} named `{node.name}` that {_first_sentence(docstring)}"
        else:
            kind = "function"
            instruction = f"Write a Python {kind} named `{node.name}` that {_first_sentence(docstring)}"

        # Add parameter info for functions
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            params = _format_params(node)
            if params:
                instruction += f"\n\nSignature: `{params}`"
            # Add full docstring if it has more detail
            if len(docstring.strip().splitlines()) > 1:
                instruction += f"\n\nDocstring:\n```\n{docstring.strip()}\n```"

        samples.append({
            "instruction": instruction,
            "response": node_source,
            "metadata": {
                "source": source,
                "file": str(filepath.relative_to(SRC_ROOT)) if filepath.is_relative_to(SRC_ROOT) else str(filepath),
                "kind": kind,
                "name": node.name,
            },
        })

    return samples


def _first_sentence(docstring: str) -> str:
    """Extract first sentence from a docstring, lowercased for instruction flow."""
    first_line = docstring.strip().split("\n")[0].strip()
    # Remove trailing period for natural flow in "that ..." construction
    if first_line.endswith("."):
        first_line = first_line[:-1]
    # Lowercase first char for "that <sentence>" grammar
    if first_line and first_line[0].isupper():
        first_line = first_line[0].lower() + first_line[1:]
    return first_line


def _format_params(node: ast.FunctionDef) -> str:
    """Format function signature with type annotations."""
    parts = []
    args = node.args

    # Regular args
    defaults_offset = len(args.args) - len(args.defaults)
    for i, arg in enumerate(args.args):
        if arg.arg == "self" or arg.arg == "cls":
            continue
        p = arg.arg
        if arg.annotation:
            p += f": {ast.unparse(arg.annotation)}"
        if i >= defaults_offset:
            default = args.defaults[i - defaults_offset]
            p += f" = {ast.unparse(default)}"
        parts.append(p)

    # *args
    if args.vararg:
        p = f"*{args.vararg.arg}"
        if args.vararg.annotation:
            p += f": {ast.unparse(args.vararg.annotation)}"
        parts.append(p)

    # **kwargs
    if args.kwarg:
        p = f"**{args.kwarg.arg}"
        if args.kwarg.annotation:
            p += f": {ast.unparse(args.kwarg.annotation)}"
        parts.append(p)

    sig = f"def {node.name}({', '.join(parts)})"
    if node.returns:
        sig += f" -> {ast.unparse(node.returns)}"
    return sig


def collect_python_files(dirs: list[Path]) -> list[Path]:
    """Recursively collect all .py files from directories."""
    files = []
    for d in dirs:
        if not d.exists():
            print(f"  warning: {d} does not exist, skipping", file=sys.stderr)
            continue
        for root, _, filenames in os.walk(d):
            for fn in sorted(filenames):
                if fn.endswith(".py") and not fn.startswith("__"):
                    files.append(Path(root) / fn)
    return files


def main():
    parser = argparse.ArgumentParser(description="Extract instruct pairs from ground truth corpora")
    parser.add_argument("--output", "-o", default="data/instruct-corpus.jsonl",
                        help="Output JSONL path (default: data/instruct-corpus.jsonl)")
    parser.add_argument("--corpus", "-c", nargs="*", choices=list(CORPORA.keys()),
                        help="Which corpora to include (default: all)")
    parser.add_argument("--min-response-lines", type=int, default=3,
                        help="Minimum lines in response to include (default: 3)")
    parser.add_argument("--max-response-lines", type=int, default=200,
                        help="Maximum lines in response to include (default: 200)")
    parser.add_argument("--stats", action="store_true",
                        help="Print corpus statistics and exit")
    args = parser.parse_args()

    selected = args.corpus or list(CORPORA.keys())
    all_samples = []

    for corpus_name in selected:
        dirs = CORPORA[corpus_name]
        files = collect_python_files(dirs)
        print(f"  {corpus_name}: {len(files)} Python files", file=sys.stderr)

        corpus_samples = []
        for f in files:
            samples = extract_functions_from_file(f, corpus_name)
            corpus_samples.extend(samples)

        # Filter by response length
        filtered = []
        for s in corpus_samples:
            nlines = s["response"].count("\n") + 1
            if args.min_response_lines <= nlines <= args.max_response_lines:
                filtered.append(s)

        print(f"  {corpus_name}: {len(corpus_samples)} raw → {len(filtered)} after length filter",
              file=sys.stderr)
        all_samples.extend(filtered)

    if args.stats:
        _print_stats(all_samples)
        return

    # Deduplicate by (name, source)
    seen = set()
    deduped = []
    for s in all_samples:
        key = (s["metadata"]["name"], s["metadata"]["source"], s["metadata"]["file"])
        if key not in seen:
            seen.add(key)
            deduped.append(s)

    print(f"\n  Total: {len(deduped)} instruction pairs (after dedup from {len(all_samples)})",
          file=sys.stderr)

    # Write output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in deduped:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"  Written to: {out_path}", file=sys.stderr)


def _print_stats(samples: list[dict]):
    """Print corpus statistics."""
    from collections import Counter
    sources = Counter(s["metadata"]["source"] for s in samples)
    kinds = Counter(s["metadata"]["kind"] for s in samples)
    lengths = [len(s["response"]) for s in samples]

    print(f"\nCorpus Statistics")
    print(f"{'─' * 50}")
    print(f"  Total samples: {len(samples)}")
    print(f"\n  By source:")
    for src, count in sources.most_common():
        print(f"    {src}: {count}")
    print(f"\n  By kind:")
    for kind, count in kinds.most_common():
        print(f"    {kind}: {count}")
    if lengths:
        print(f"\n  Response length (chars):")
        print(f"    min: {min(lengths)}")
        print(f"    max: {max(lengths)}")
        print(f"    avg: {sum(lengths) // len(lengths)}")
        print(f"    median: {sorted(lengths)[len(lengths) // 2]}")


if __name__ == "__main__":
    main()
