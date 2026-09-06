"""The invariants that decide whether the model comparison means anything.

A held-out comparison is only worth the split behind it, and a label is only worth
the thing it marks. Both were wrong here at first, in ways that produced a large,
entirely fake win for the learned models, so both are pinned down.

These import nothing from sklearn, mlflow or pandas: the harness that generates and
splits the data is stdlib, so CI can check the integrity of the experiment without
installing a training stack to do it.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import episodes as ep_mod
from experiments import features as feat

SEED = 20260905


@pytest.fixture(scope="module")
def eps():
    return ep_mod.generate(24, seed=SEED)


# ── split integrity ────────────────────────────────────────────────────────────
def test_no_run_appears_on_both_sides_of_the_split(eps):
    tr, va, te = ep_mod.split_by_episode(eps)
    ids = [{e["run_id"] for e in part} for part in (tr, va, te)]

    assert ids[0] and ids[1] and ids[2]
    assert not ids[0] & ids[1]
    assert not ids[0] & ids[2]
    assert not ids[1] & ids[2]
    assert sum(len(i) for i in ids) == len(eps)


def test_no_metric_row_is_shared_between_train_and_test(eps):
    """Windows inside one run overlap, so a split on windows would leak.

    Consecutive windows share all but one row. Splitting on them puts near-copies of
    the same moment on both sides and scores memorisation as generalisation, which is
    exactly why the split is on whole runs.
    """
    tr, _, te = ep_mod.split_by_episode(eps)

    def rows(part):
        return {(r["run_id"], r["step"]) for e in part for r in e["rows"]}

    assert not rows(tr) & rows(te)


def test_windows_never_span_two_runs(eps):
    for _, _, run_id, w in feat.windows(eps):
        assert {r["run_id"] for r in w} == {run_id}


def test_generation_is_reproducible_from_the_seed(eps):
    assert ep_mod.dataset_hash(ep_mod.generate(24, seed=SEED)) == ep_mod.dataset_hash(eps)
    assert ep_mod.dataset_hash(ep_mod.generate(24, seed=SEED + 1)) != ep_mod.dataset_hash(eps)


# ── label integrity ────────────────────────────────────────────────────────────
def test_a_fault_never_feeds_back_into_the_underlying_trajectory():
    """The regression test for the bug that invalidated the first comparison.

    Applying the spike multiplier to the running loss variable compounded it over
    three rows to roughly 200x and left it there for the rest of the episode. Every
    later row stayed labelled faulty while carrying a level no healthy run reaches, so
    the label became a property of the episode rather than of the moment, and a
    classifier could score it without detecting anything.

    After a spike ends, the run must be back in the same band as a clean one.
    """
    eps = ep_mod.generate(20, seed=SEED)
    spike = next(e for e in eps if e["fault"] == "loss_spike")
    clean = next(e for e in eps if e["fault"] == "none")

    onset = next(i for i, r in enumerate(spike["rows"]) if r["fault"] != "none")
    during = max(r["train_loss"] for r in spike["rows"][onset:onset + ep_mod.SPIKE_ROWS])
    after = [r["train_loss"] for r in spike["rows"][onset + ep_mod.SPIKE_ROWS:]]
    clean_after = [r["train_loss"] for r in clean["rows"][onset + ep_mod.SPIKE_ROWS:]]

    assert during > 3 * max(after)                       # the spike was real
    assert max(after) < 2 * max(clean_after)             # and the run recovered


def test_a_transient_fault_is_labelled_only_while_it_is_present():
    eps = ep_mod.generate(20, seed=SEED)
    spike = next(e for e in eps if e["fault"] == "loss_spike")
    flagged = [i for i, r in enumerate(spike["rows"]) if r["fault"] != "none"]

    assert len(flagged) == ep_mod.SPIKE_ROWS
    assert flagged == list(range(flagged[0], flagged[0] + ep_mod.SPIKE_ROWS))


def test_a_sustained_fault_stays_labelled_to_the_end_of_the_run():
    """Unlike a spike, these really are ongoing states, so the label is ongoing too."""
    eps = ep_mod.generate(20, seed=SEED)
    for kind in ("grad_explosion", "val_plateau", "overfitting"):
        rows = next(e for e in eps if e["fault"] == kind)["rows"]
        flagged = [i for i, r in enumerate(rows) if r["fault"] != "none"]
        assert flagged, kind
        assert flagged == list(range(flagged[0], len(rows))), kind


def test_clean_episodes_carry_no_positive_label():
    eps = ep_mod.generate(20, seed=SEED)
    for e in eps:
        if e["fault"] == "none":
            assert all(r["fault"] == "none" for r in e["rows"])


def test_faults_start_after_a_full_reference_window():
    """A window at the onset needs healthy history to be judged against.

    Without it the z-score has nothing meaningful to compare to and the first
    detection is decided by where the episode happened to begin.
    """
    eps = ep_mod.generate(20, seed=SEED)
    for e in eps:
        flagged = [i for i, r in enumerate(e["rows"]) if r["fault"] != "none"]
        if flagged:
            assert flagged[0] >= feat.WINDOW, e["fault"]
