"""The rerun server and the trainer must agree on how the trainer is launched.

They did not. server.py spawned `train.py --max-steps 100`; train.py read
`int(sys.argv[1])`. Every rerun the agent ever requested died in

    ValueError: invalid literal for int() with base 10: '--max-steps'

before a single optimizer step, and the exit status was discarded inside a daemon
thread that the HTTP handler had already answered ahead of. The agent then recorded
the run as fixed.

The two sides now build and parse through training_job/cli.py, so the contract is
stated once. These tests are the round trip, plus the argument handling that has to
keep working for a launch to be observable at all. They import nothing from the
training stack on purpose - the contract is checkable without torch installed, which
is what lets CI catch a mismatch.
"""

import subprocess
import sys
from pathlib import Path

import pytest

TRAINING_JOB = Path(__file__).resolve().parents[1] / "training_job"
sys.path.insert(0, str(TRAINING_JOB))

from cli import build_training_argv, parse_args


def test_the_command_the_server_builds_is_one_the_trainer_accepts():
    """The round trip that was broken: build it, then parse it.

    argv[0] is the script name, so what the trainer sees as arguments is argv[1:].
    """
    argv = build_training_argv(max_steps=100)

    assert argv[0] == "train.py"
    assert parse_args(argv[1:]).max_steps == 100


def test_the_old_positional_form_is_rejected_rather_than_misread():
    """`train.py 100` must fail loudly instead of quietly meaning something else."""
    with pytest.raises(SystemExit):
        parse_args(["100"])


def test_omitting_max_steps_means_run_the_configured_epochs():
    assert build_training_argv() == ["train.py"]
    assert parse_args([]).max_steps is None


def test_a_non_integer_step_count_fails_at_parse_time():
    with pytest.raises(SystemExit):
        parse_args(["--max-steps", "not-a-number"])


def test_the_contract_is_checkable_without_the_training_stack():
    """cli.py must not drag in torch.

    If parsing needed the training stack, a launch-contract mismatch could only be
    caught by running a real training job, which is exactly why this one survived.
    """
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, %r); import cli; "
         "assert 'torch' not in sys.modules; print('ok')" % str(TRAINING_JOB)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
