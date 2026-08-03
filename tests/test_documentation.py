from pathlib import Path

from tmd_policy.cli import build_parser

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_READMES = [
    "README.md",
    "configs/README.md",
    "scripts/README.md",
    "tests/README.md",
    "experiments/README.md",
    "experiments/motivation/README.md",
    "src/tmd_policy/models/README.md",
    "src/tmd_policy/data/README.md",
    "src/tmd_policy/rollout/README.md",
    "src/tmd_policy/teacher/README.md",
    "src/tmd_policy/training/README.md",
    "src/tmd_policy/compatibility/README.md",
    "src/tmd_policy/evaluation/README.md",
]


def test_required_readmes_exist_and_cover_every_python_file_in_module():
    for relative in REQUIRED_READMES:
        path = PROJECT_ROOT / relative
        assert path.is_file() and path.stat().st_size > 100
        if path.parent.name in {
            "models",
            "data",
            "rollout",
            "teacher",
            "training",
            "compatibility",
            "evaluation",
            "motivation",
        }:
            contents = path.read_text(encoding="utf-8")
            for python_file in path.parent.glob("*.py"):
                assert python_file.name in contents, f"{path} omits {python_file.name}"


def test_documentation_contains_all_nine_required_architecture_flows():
    markdown = "\n".join(
        path.read_text(encoding="utf-8") for path in PROJECT_ROOT.rglob("*.md")
    )
    assert markdown.count("```mermaid") >= 9


def test_cli_exposes_every_required_research_command():
    parser = build_parser()
    subparsers = next(action for action in parser._actions if action.dest == "command")
    required = {
        "audit",
        "inspect",
        "build-expert",
        "synthetic-smoke",
        "collect-rollouts",
        "train-discriminator",
        "evaluate-discriminator",
        "plot-motivation",
        "train-expert",
        "query-teacher",
        "distill",
        "evaluate-policy",
        "run-experiment",
    }
    assert required <= set(subparsers.choices)
