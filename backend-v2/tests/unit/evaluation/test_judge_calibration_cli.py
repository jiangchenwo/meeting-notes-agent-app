import json
from pathlib import Path
import subprocess
import sys


def test_qualification_cli_preflight_is_offline_and_redacted(tmp_path: Path) -> None:
    backend = Path(__file__).resolve().parents[3]
    config = tmp_path / "judge.json"
    config.write_text(json.dumps({
        "schema_version": "judge-settings-v1",
        "provider": "disabled",
        "model": None,
        "base_url": None,
        "timeout_seconds": 30,
        "max_cost_usd": 0,
        "input_cost_per_million": 0,
        "output_cost_per_million": 0,
        "temperature": 0,
        "rubric": "issues-v1",
    }))
    env_file = tmp_path / ".env"
    env_file.write_text("\n".join((
        f"NOTES_EVAL_JUDGE_CONFIG_FILE={config}",
        "NOTES_EVAL_JUDGE_PROVIDER=openai_compatible",
        "NOTES_EVAL_JUDGE_MODEL=scripted-model",
        "NOTES_EVAL_JUDGE_BASE_URL=https://example.test/v1",
        "NOTES_EVAL_JUDGE_MAX_COST_USD=1.00",
        "NOTES_EVAL_JUDGE_INPUT_COST_PER_MILLION=0.25",
        "NOTES_EVAL_JUDGE_OUTPUT_COST_PER_MILLION=1.50",
        "NOTES_EVAL_JUDGE_API_TOKEN=qualification-secret",
    )))

    result = subprocess.run(
        [
            sys.executable,
            str(backend / "scripts/qualify_remote_judge.py"),
            "--suite", str(backend / "tests/fixtures/evaluation/judge-calibration.json"),
            "--env-file", str(env_file),
            "--preflight-only",
        ],
        cwd=backend,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "requests=60" in result.stdout
    assert "qualification-secret" not in result.stdout + result.stderr
