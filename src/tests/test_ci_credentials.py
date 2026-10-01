import json
import os
import subprocess
from pathlib import Path
from textwrap import dedent


def test_e2e_credentials_preserve_provider_keys():
    workflow = Path(__file__).resolve().parents[2] / ".github/workflows/ci.yaml"
    step = workflow.read_text().split("      - name: Run E2E tests\n", 1)[1]
    script = dedent(
        step.split("        run: |\n", 1)[1].split("          timeout ", 1)[0]
    )
    environment = {
        "OPENAI_API_KEY": "github-openai",
        "ANTHROPIC_API_KEY": "github-anthropic",
        "GOOGLE_API_KEY": "github-google",
        "BASE_URL": "https://api.example.test",
        "SECRETS_PATH": "prod/api-keys/e2e-tests",
        "TEST_SECRET": json.dumps(
            {
                "OPENAI_API_KEY": "stale-openai",
                "ANTHROPIC_API_KEY": "stale-anthropic",
                "GEMINI_API_KEY": "stale-google",
                "JUDGEVAL_GH_JUDGMENT_API_KEY": "judgment-test-key",
                "JUDGEVAL_GH_JUDGMENT_ORG_ID": "judgment-test-org",
            }
        ),
    }
    result = subprocess.run(
        [
            "bash",
            "-e",
            "-c",
            'aws() { printf "%s" "$TEST_SECRET"; }\n'
            + script
            + '\nprintf "%s\\n" "$OPENAI_API_KEY" "$ANTHROPIC_API_KEY" "$GOOGLE_API_KEY" "$JUDGMENT_API_KEY" "$JUDGMENT_ORG_ID" "$JUDGMENT_API_URL"',
        ],
        env={**os.environ, **environment},
        capture_output=True,
        text=True,
    )
    assert (result.returncode, result.stdout, result.stderr) == (
        0,
        "github-openai\ngithub-anthropic\ngithub-google\njudgment-test-key\njudgment-test-org\nhttps://api.example.test\n",
        "",
    )
