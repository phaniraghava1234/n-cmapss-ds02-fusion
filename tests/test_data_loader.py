"""Phase 1 acceptance gate.

Asserts the loaded data reproduces target paper Table 1 (p. 7) exactly, and
that the variable groups match target paper Tables 2-4 (p. 8).

PROJECT_PLAN.md Sec. 4 makes any mismatch here a blocker: if these fail, the
data is not what the paper used and nothing downstream is comparable.

Run:  pytest tests/test_data_loader.py -v
  or: python tests/test_data_loader.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import (  # noqa: E402
    BASELINE_INPUT, DECIMATION_FACTOR, THETA_PAPER, W_VARS, X_S_PAPER,
    X_V_PAPER, decimation_index, load_processed, unit_summary,
)

import numpy as np  # noqa: E402

#: Target paper Table 1 (p. 7): unit -> (m_i, t_s, t_EOL, failure mode).
#: m_i is quoted in millions to three decimals, i.e. rounded to the nearest
#: 1000 samples, so a reproduction is correct if it lands within +/-500.
PAPER_TABLE_1 = {
    2:  (85_000, 17, 75, "HPT"),
    5:  (103_000, 17, 89, "HPT"),
    10: (95_000, 17, 82, "HPT"),
    16: (77_000, 16, 63, "HPT+LPT"),
    18: (89_000, 17, 71, "HPT+LPT"),
    20: (77_000, 17, 66, "HPT+LPT"),
    11: (66_000, 19, 59, "HPT+LPT"),
    14: (16_000, 36, 76, "HPT+LPT"),
    15: (43_000, 24, 67, "HPT+LPT"),
}

M_I_TOLERANCE = 500  # half of the paper's rounding unit

DEV_UNITS = [2, 5, 10, 16, 18, 20]
TEST_UNITS = [11, 14, 15]

#: Sums of the Table 1 m_i column. Sec. 4.1 (p. 6) quotes these as 0.53M and
#: 0.12M, but the table entries are the tighter figures, so we test against
#: those instead of the rounded prose values.
PAPER_DEV_SAMPLES = 526_000
PAPER_TEST_SAMPLES = 125_000


@pytest.fixture(scope="module")
def summaries():
    return {split: unit_summary(load_processed(split))
            for split in ("dev", "test")}


# --------------------------------------------------------- Table 1 agreement

@pytest.mark.parametrize("split,expected", [("dev", DEV_UNITS),
                                            ("test", TEST_UNITS)])
def test_unit_split(summaries, split, expected):
    """Unit ids match the paper's development / test partition."""
    assert sorted(summaries[split]["unit"]) == expected


@pytest.mark.parametrize("unit", sorted(PAPER_TABLE_1))
def test_sample_count(summaries, unit):
    """m_i reproduces the paper within its own rounding precision.

    This is the test that proves the 1 Hz -> 0.1 Hz reduction was right. At the
    released sampling it would miss by a factor of ten.
    """
    combined = summaries["dev"] if unit in DEV_UNITS else summaries["test"]
    row = combined.loc[combined["unit"] == unit].iloc[0]
    expected = PAPER_TABLE_1[unit][0]
    assert abs(int(row["m_i"]) - expected) <= M_I_TOLERANCE, (
        f"unit {unit}: got {int(row['m_i']):,}, paper {expected:,}"
    )


@pytest.mark.parametrize("unit", sorted(PAPER_TABLE_1))
def test_transition_cycle(summaries, unit):
    """t_s matches exactly. Sampling-rate independent."""
    combined = summaries["dev"] if unit in DEV_UNITS else summaries["test"]
    row = combined.loc[combined["unit"] == unit].iloc[0]
    assert int(row["t_s"]) == PAPER_TABLE_1[unit][1]


@pytest.mark.parametrize("unit", sorted(PAPER_TABLE_1))
def test_end_of_life_cycle(summaries, unit):
    """t_EOL matches exactly.

    Cycle counts do not depend on sampling rate, so this is the fingerprint
    identifying our release as the same simulation the paper used.
    """
    combined = summaries["dev"] if unit in DEV_UNITS else summaries["test"]
    row = combined.loc[combined["unit"] == unit].iloc[0]
    assert int(row["t_EOL"]) == PAPER_TABLE_1[unit][2]


def test_split_totals(summaries):
    """Split totals reproduce the sums of Table 1 to within 1%."""
    dev = int(summaries["dev"]["m_i"].sum())
    test = int(summaries["test"]["m_i"].sum())
    assert abs(dev - PAPER_DEV_SAMPLES) < 0.01 * PAPER_DEV_SAMPLES
    assert abs(test - PAPER_TEST_SAMPLES) < 0.01 * PAPER_TEST_SAMPLES


# ------------------------------------------------------ variable definitions

def test_paper_columns_present():
    """Every signal Tables 2-4 require exists in the loaded data."""
    frame = load_processed("test")
    missing = [c for c in W_VARS + X_S_PAPER + X_V_PAPER + THETA_PAPER
               if c not in frame.columns]
    assert not missing, f"missing from release: {missing}"


def test_variable_group_sizes():
    """Group sizes match target paper Tables 2-4."""
    assert len(W_VARS) == 4
    assert len(X_S_PAPER) == 16      # Table 2 minus the four in w
    assert len(X_V_PAPER) == 11      # Table 3
    assert len(THETA_PAPER) == 3     # Table 4


def test_model_input_dimensions():
    """Sec. 4.3 (p. 7) states 20 baseline inputs and 50 hybrid inputs."""
    assert len(BASELINE_INPUT) == 20
    hybrid = len(W_VARS) + len(X_S_PAPER) + len(X_S_PAPER) + \
        len(X_V_PAPER) + len(THETA_PAPER)
    assert hybrid == 50


def test_phi_excluded():
    """phi is in the release (dataset paper Table 6) but not in the paper's x_v.

    It is the ratio of fuel flow to Ps30 -- derived from two values already
    present -- which is why target paper Table 3 lists 11 rather than 12.
    """
    frame = load_processed("test")
    assert "phi" in frame.columns
    assert "phi" not in X_V_PAPER


def test_t40_p30_promoted():
    """T40 and P30 move from x_v to x_s. Locked decision V1, Deviation 2."""
    assert "T40" in X_S_PAPER and "P30" in X_S_PAPER
    assert "T40" not in X_V_PAPER and "P30" not in X_V_PAPER


# ------------------------------------------------------- decimation mechanics

def test_decimation_respects_cycle_boundaries():
    """Cycles of length 25 and 13 yield 3 and 2 samples, not a global stride.

    A global stride would carry the remainder across the boundary and pick up
    the wrong rows in the second cycle.
    """
    unit = np.array([2] * 25 + [2] * 13)
    cycle = np.array([1] * 25 + [2] * 13)
    index = decimation_index(unit, cycle, factor=10, offset=0)
    assert index.tolist() == [0, 10, 20, 25, 35]


def test_decimation_rejects_noncontiguous_cycles():
    """An interleaved cycle is a malformed file, not something to silently fix."""
    unit = np.array([2, 2, 2, 2])
    cycle = np.array([1, 2, 1, 2])
    with pytest.raises(ValueError, match="not contiguous"):
        decimation_index(unit, cycle, factor=1)


def test_decimation_factor_is_ten():
    """1 Hz release -> 0.1 Hz paper sampling."""
    assert DECIMATION_FACTOR == 10


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
