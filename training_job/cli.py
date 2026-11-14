"""The launch contract between the rerun server and the trainer.

server.py spawns the trainer as a subprocess and train.py parses what it gets. The
two used to state that contract separately, and they disagreed: server.py sent
`train.py --max-steps 100` while train.py read `int(sys.argv[1])`, so every launch
died in `ValueError: invalid literal for int() with base 10: '--max-steps'` before a
single step ran. Nothing upstream noticed, because the exit status was discarded in a
thread the HTTP handler had already returned ahead of.

Both sides now build and parse through this module, so the two halves of the contract
cannot drift apart again without failing here first. It deliberately imports nothing
heavier than argparse: a wrong flag should be rejected in milliseconds, and the
contract stays testable on a machine with no training stack installed.
"""

from __future__ import annotations

import argparse

TRAINER_SCRIPT = "train.py"


def build_parser():
    parser = argparse.ArgumentParser(prog=TRAINER_SCRIPT)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="stop after this many optimizer steps; omit to run the configured epochs",
    )
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def build_training_argv(max_steps=None):
    """The argument vector to launch the trainer with, without the interpreter."""
    argv = [TRAINER_SCRIPT]
    if max_steps is not None:
        argv += ["--max-steps", str(max_steps)]
    return argv
