"""run_demo.py — One-command tour of the whole repo (offline).

Runs: synthetic data -> indexing -> a sweep -> report -> MCP agent demo.
No API keys required.
"""

import os
import subprocess
import sys

# Always run from the repo root so relative paths resolve.
os.chdir(os.path.dirname(os.path.abspath(__file__)))


STEPS = [
    ("Generate synthetic data", [sys.executable, "scripts/generate_synthetic_data.py"]),
    ("Index documents + query", [sys.executable, "scripts/index_documents.py"]),
    ("Run hyperparameter sweep (20 experiments)",
     [sys.executable, "scripts/run_sweep.py", "20"]),
    ("Sweep report", [sys.executable, "-m", "evaluation.report_generator"]),
    ("Case studies (RAG + MCP)", [sys.executable, "scripts/benchmark.py"]),
]


def main() -> None:
    for title, cmd in STEPS:
        print("\n" + "#" * 70)
        print(f"# {title}")
        print("#" * 70)
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"Step failed: {title}")
            sys.exit(rc)
    print("\nAll steps completed.")


if __name__ == "__main__":
    main()
