#!/usr/bin/env python3
"""Assemble a test file from benchmark problem + model completion.

Usage:
    assemble-test.py <benchmark> <problem.json> <completion.py> <output.py>

Reads a single JSONL problem line, a completion file, and writes a
combined test file that can be executed with `python3 output.py`.
"""

import json
import sys


def strip_markdown_fences(text: str) -> str:
    """Remove ```python ... ``` wrapping from model output."""
    lines = text.strip().split("\n")
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


def assemble_humaneval(problem: dict, completion: str) -> str:
    """Assemble HumanEval test: prompt + completion + test harness + check()."""
    prompt = problem["prompt"]
    entry_point = problem["entry_point"]
    test_harness = problem.get("test", "")

    completion = strip_markdown_fences(completion)

    # If the completion contains the full function definition, use it directly
    if f"def {entry_point}" in completion:
        # Extract imports from prompt if any
        prompt_lines = prompt.split("\n")
        imports = [l for l in prompt_lines if l.startswith(("import ", "from "))]
        code = "\n".join(imports) + "\n\n" + completion if imports else completion
    else:
        # Completion is just the function body — prepend the prompt
        code = prompt + completion

    code += "\n"
    code += test_harness
    code += f"\ncheck({entry_point})\n"
    return code


def assemble_mbpp(problem: dict, completion: str) -> str:
    """Assemble MBPP test: completion + test assertions."""
    completion = strip_markdown_fences(completion)
    test_list = problem.get("test_list", [])
    tests = "\n".join(test_list) if test_list else problem.get("test", "")

    return completion + "\n\n" + tests + "\n"


def main():
    if len(sys.argv) != 5:
        print(f"Usage: {sys.argv[0]} <benchmark> <problem.json> <completion.py> <output.py>")
        sys.exit(1)

    benchmark = sys.argv[1]
    problem_file = sys.argv[2]
    completion_file = sys.argv[3]
    output_file = sys.argv[4]

    with open(problem_file) as f:
        problem = json.loads(f.read())

    with open(completion_file) as f:
        completion = f.read()

    if benchmark == "humaneval":
        code = assemble_humaneval(problem, completion)
    elif benchmark == "mbpp":
        code = assemble_mbpp(problem, completion)
    else:
        # Generic: completion + test
        completion = strip_markdown_fences(completion)
        test = problem.get("test", "")
        code = completion + "\n\n" + test + "\n"

    with open(output_file, "w") as f:
        f.write(code)


if __name__ == "__main__":
    main()
