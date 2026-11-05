"""Phase 1 -- N-CMAPSS DS02 ingestion.

Reads the DS02 HDF5 release, reduces it from the released 1 Hz sampling to the
0.1 Hz sampling the target paper worked at, and writes Parquet.

Paper references
----------------
Target paper -- Arias Chao M, Kulkarni C, Goebel K, Fink O (2022), "Fusing
physics-based and deep learning models for prognostics", Reliability
Engineering & System Safety 217, 107961.

    Sec. 4.1 (p. 6)   sampling rate 0.1 Hz; 0.53M development, 0.12M test
    Table 1  (p. 7)   per-unit m_i, t_s, t_EOL, failure mode
    Table 2  (p. 8)   condition monitoring signals [w, x_s] -- 20 entries
    Table 3  (p. 8)   virtual sensors x_v -- 11 entries
    Table 4  (p. 8)   health parameters theta -- 3 entries

Dataset paper -- Arias Chao M et al. (2021), Data 6(1):5.

    Table 2  (p. 6)   DS02 size 6.5M  <- matches the release we hold
    Table 5  (p. 7)   x_s as released -- 14 entries
    Table 6  (p. 7)   x_v as released -- 14 entries, including phi

Why the release and the paper disagree on size
----------------------------------------------
The target paper states 0.1 Hz and 0.65M total samples. The release holds
6.5M samples, exactly 10x more, and the dataset paper's Table 2 confirms 6.5M
for DS02. Target paper Sec. 4.1 points to reference [23] -- the dataset paper --
for "a more recent version of this dataset", so the paper used an earlier
release. Unit ids, cycle counts, transition cycles, flight classes and failure
modes all match exactly between the paper's Table 1 and this file, so the
underlying simulation is the same; only the sampling differs.

We therefore keep every 10th sample within each flight cycle. See
REPLICATION_NOTES.md deviations 4 (sampling rate), 5 (dataset version) and
6 (phi exclusion).

Naming
------
Sensor names are kept verbatim from the paper and the HDF5 file (T24, Ps30,
alt, ...) because they are the paper's own symbols. Everything else is
snake_case per locked decision D2.
"""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# --------------------------------------------------------------------- paths

DS02_FILENAME = "N-CMAPSS_DS02-006.h5"
PROCESSED_DIRNAME = "processed"


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    """Data directory. Override with N_CMAPSS_DATA for Colab or a moved copy."""
    override = os.environ.get("N_CMAPSS_DATA")
    return Path(override) if override else repo_root() / "data"


def ds02_path() -> Path:
    return data_dir() / DS02_FILENAME


def processed_dir() -> Path:
    return data_dir() / PROCESSED_DIRNAME


# ------------------------------------------------------ paper variable groups

#: Scenario descriptors w -- target paper Table 2 rows 1-3 and 7, dataset
#: paper Table 4. Same four in both.
W_VARS = ["alt", "Mach", "TRA", "T2"]

#: Measured signals x_s -- target paper Table 2 (20 entries) minus the four in
#: W_VARS. T40 and P30 sit in X_v_var in the release; the paper treats them as
#: measured, so they are promoted here. Locked decision V1.
X_S_PAPER = [
    "Wf", "Nf", "Nc", "T24", "T30", "T40", "T48", "T50",
    "P15", "P2", "P21", "P24", "Ps30", "P30", "P40", "P50",
]

#: Virtual sensors x_v -- target paper Table 3, 11 entries. The release also
#: carries phi (fuel flow / Ps30, dataset paper Table 6 row 14), a derived
#: ratio of two values already present; the paper does not use it.
X_V_PAPER = [
    "P45", "W21", "W22", "W25", "W31", "W32", "W48", "W50",
    "SmFan", "SmLPC", "SmHPC",
]

#: Health parameters theta -- target paper Table 4. The release carries all 10
#: modifiers; DS02's two failure modes only move these three.
THETA_PAPER = ["HPT_eff_mod", "LPT_eff_mod", "LPT_flow_mod"]

#: Auxiliary columns -- dataset paper Table 8.
AUX_VARS = ["unit", "cycle", "Fc", "hs"]

RUL_VAR = "RUL"

#: Baseline input [w, x_s], n = 20 -- target paper Sec. 4.3.
BASELINE_INPUT = W_VARS + X_S_PAPER

# --------------------------------------------------------------- downsampling

#: 1 Hz release -> 0.1 Hz paper sampling.
DECIMATION_FACTOR = 10

#: Which sample of each group of ten to keep. The old release's phase is
#: unknowable, so we take the first. Recorded in REPLICATION_NOTES.md.
DECIMATION_OFFSET = 0


#: Rows per sequential read. Keeps peak memory near this many rows rather
#: than the whole dataset.
READ_CHUNK_ROWS = 500_000


def _var_names(hdf: h5py.File, key: str) -> list[str]:
    return [s.decode() if isinstance(s, bytes) else str(s)
            for s in hdf[key][()].ravel()]


def _gather(dset: h5py.Dataset, index: np.ndarray, label: str,
            chunk: int = READ_CHUNK_ROWS) -> np.ndarray:
    """Collect `index` rows from `dset`, reading sequentially in chunks.

    Reading a whole dataset and then fancy-indexing needs the full array
    resident -- 562 MB for one DS02 block, and more again while the indexed
    copy is built. Sequential chunks hold roughly `chunk` rows instead, let the
    OS read ahead, and report progress, which matters when the file is served
    through OneDrive rather than straight off disk.

    `index` must be ascending; decimation_index produces it that way.
    """
    n_rows = dset.shape[0]
    out = np.empty((index.size,) + dset.shape[1:], dtype=dset.dtype)
    for start in range(0, n_rows, chunk):
        stop = min(start + chunk, n_rows)
        lo = int(np.searchsorted(index, start, side="left"))
        hi = int(np.searchsorted(index, stop, side="left"))
        if hi > lo:
            out[lo:hi] = dset[start:stop][index[lo:hi] - start]
        print(f"    {label:<12} {100 * stop / n_rows:5.1f}%", end="\r",
              flush=True)
    print(f"    {label:<12} {index.size:>9,} rows", flush=True)
    return out


def decimation_index(unit: np.ndarray, cycle: np.ndarray,
                     factor: int = DECIMATION_FACTOR,
                     offset: int = DECIMATION_OFFSET) -> np.ndarray:
    """Row positions keeping every `factor`-th sample within each flight cycle.

    Decimating across the whole array would drift over cycle boundaries, since
    cycle lengths are not multiples of `factor`, and would blend the end of one
    flight into the start of the next. Grouping by (unit, cycle) prevents that.

    Rows of a cycle are contiguous and time-ordered in the release. That is
    checked rather than assumed -- it is an external file.
    """
    key = unit.astype(np.int64) * 100_000 + cycle.astype(np.int64)
    change = np.flatnonzero(key[1:] != key[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [key.size]))

    if starts.size != np.unique(key).size:
        raise ValueError(
            f"(unit, cycle) rows are not contiguous: {starts.size} runs for "
            f"{np.unique(key).size} unique pairs. Decimation assumes each "
            f"flight cycle occupies one contiguous block."
        )

    return np.concatenate([np.arange(s + offset, e, factor)
                           for s, e in zip(starts, ends)])


# -------------------------------------------------------------------- loading

def load_split(split: str, path: Path | None = None,
               decimate: bool = True) -> pd.DataFrame:
    """Load one split of DS02 as a DataFrame.

    Parameters
    ----------
    split : "dev" or "test"
        The paper's development set (units 2, 5, 10, 16, 18, 20) or test set
        (units 11, 14, 15). Target paper Table 1.
    decimate : bool
        Reduce 1 Hz to 0.1 Hz. False returns the release untouched, which is
        the v3 sampling-rate ablation in PROJECT_PLAN.md Sec. 16.

    Returns
    -------
    DataFrame with every column the release carries -- all 14 x_s, all 14 x_v
    including phi, all 10 theta -- not just the paper's subset. The paper's
    selection happens at model-input time through the constants above, so the
    stored data stays a faithful decimated copy of DS02.
    """
    if split not in ("dev", "test"):
        raise ValueError(f"split must be 'dev' or 'test', got {split!r}")

    path = path or ds02_path()
    if not path.exists():
        raise FileNotFoundError(
            f"DS02 not found at {path}. Set N_CMAPSS_DATA to the directory "
            f"holding {DS02_FILENAME}."
        )

    with h5py.File(path, "r") as hdf:
        groups = [
            ("W", _var_names(hdf, "W_var")),
            ("X_s", _var_names(hdf, "X_s_var")),
            ("X_v", _var_names(hdf, "X_v_var")),
            ("T", _var_names(hdf, "T_var")),
        ]

        aux = hdf[f"A_{split}"][()]
        index = (decimation_index(aux[:, 0], aux[:, 1]) if decimate
                 else np.arange(aux.shape[0]))
        print(f"  {split}: {aux.shape[0]:,} rows -> {index.size:,} kept",
              flush=True)

        columns: dict[str, np.ndarray] = {}
        for name, value in zip(AUX_VARS, aux[index].T):
            columns[name] = np.ascontiguousarray(value)
        del aux

        for prefix, names in groups:
            block = _gather(hdf[f"{prefix}_{split}"], index,
                            f"{prefix}_{split}")
            # Copy each column so the gathered block can be released; a plain
            # slice would be a view and would pin the whole block in memory.
            for j, name in enumerate(names):
                columns[name] = np.ascontiguousarray(block[:, j])
            del block

        columns[RUL_VAR] = _gather(hdf[f"Y_{split}"], index,
                                   f"Y_{split}").ravel()

    frame = pd.DataFrame(columns)
    frame["unit"] = frame["unit"].astype(np.int16)
    frame["cycle"] = frame["cycle"].astype(np.int16)
    frame["Fc"] = frame["Fc"].astype(np.int8)
    frame["hs"] = frame["hs"].astype(np.int8)
    frame[RUL_VAR] = frame[RUL_VAR].astype(np.int16)
    return frame


def build_processed(decimate: bool = True,
                    out_dir: Path | None = None) -> dict[str, Path]:
    """Write both splits to Parquet. Returns the paths written."""
    out_dir = out_dir or processed_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    written = {}
    for split in ("dev", "test"):
        frame = load_split(split, decimate=decimate)
        target = out_dir / f"ds02_{split}.parquet"
        frame.to_parquet(target, index=False)
        written[split] = target
        print(f"{split:>4}: {len(frame):>9,} rows x {frame.shape[1]} cols "
              f"-> {target.name} "
              f"({target.stat().st_size / 1024**2:.1f} MB)")
    return written


def load_processed(split: str, out_dir: Path | None = None) -> pd.DataFrame:
    """Read a split back from Parquet."""
    out_dir = out_dir or processed_dir()
    target = out_dir / f"ds02_{split}.parquet"
    if not target.exists():
        raise FileNotFoundError(
            f"{target} missing. Run: python src/data_loader.py"
        )
    return pd.read_parquet(target)


def unit_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-unit statistics comparable to target paper Table 1.

    t_s is the first cycle whose samples are no longer flagged healthy. The
    paper's Table 1 uses the same convention; deriving it as the *last* healthy
    cycle instead gives values one lower across all nine units.
    """
    rows = []
    for unit, group in frame.groupby("unit", sort=True):
        degraded = group.loc[group["hs"] == 0, "cycle"]
        rows.append({
            "unit": int(unit),
            "Fc": int(group["Fc"].iloc[0]),
            "m_i": len(group),
            "t_s": int(degraded.min()) if len(degraded) else -1,
            "t_EOL": int(group["cycle"].max()),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(f"source : {ds02_path()}")
    print(f"target : {processed_dir()}")
    print(f"sampling: 1 Hz -> {1 / DECIMATION_FACTOR:g} Hz "
          f"(every {DECIMATION_FACTOR}th sample, offset {DECIMATION_OFFSET})")
    print()
    build_processed()
    print()
    for split in ("dev", "test"):
        print(f"--- {split} ---")
        print(unit_summary(load_processed(split)).to_string(index=False))
