import json
import subprocess
from pathlib import Path


def test_cli_generate_analyze_eval_end_to_end(tmp_path: Path):
    synth_dir = tmp_path / "synth"
    audit_path = tmp_path / "audit.json"

    generate = subprocess.run(
        [".venv/bin/partoguard", "generate", "--output-dir", str(synth_dir)],
        cwd="/root/work",
        text=True,
        capture_output=True,
        check=False,
    )
    assert generate.returncode == 0
    assert "Generated 18" in generate.stdout

    analyze = subprocess.run(
        [".venv/bin/partoguard", "analyze", str(synth_dir / "action_zone_clean_fullpage.png"), "--json-out", str(audit_path)],
        cwd="/root/work",
        text=True,
        capture_output=True,
        check=False,
    )
    assert analyze.returncode == 0
    assert "Status: action_zone" in analyze.stdout
    audit = json.loads(audit_path.read_text())
    assert audit["rule_output"]["status"] == "action_zone"
    assert "image_path" not in audit["metadata"]
    assert audit["metadata"]["input_id"]

    evaluate = subprocess.run(
        [".venv/bin/partoguard", "eval", "--synthetic-dir", str(synth_dir)],
        cwd="/root/work",
        text=True,
        capture_output=True,
        check=False,
    )
    assert evaluate.returncode == 0
    assert "Non-manual zone accuracy" in evaluate.stdout
    assert "Full-set success rate" in evaluate.stdout


def test_cli_eval_full_corpus_command():
    evaluate = subprocess.run(
        [".venv/bin/partoguard", "eval", "--corpus-dir", "data"],
        cwd="/root/work",
        text=True,
        capture_output=True,
        check=False,
    )

    assert evaluate.returncode == 0
    assert "PartoGuard corpus evaluation" in evaluate.stdout
    assert "Total images: 350" in evaluate.stdout
    assert "Blank-template manual review rate" in evaluate.stdout


def test_cli_missing_image_does_not_echo_path(tmp_path: Path):
    missing = tmp_path / "patient-name-should-not-leak.png"

    result = subprocess.run(
        [".venv/bin/partoguard", "analyze", str(missing)],
        cwd="/root/work",
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "image not found" in result.stderr
    assert "patient-name" not in result.stderr
