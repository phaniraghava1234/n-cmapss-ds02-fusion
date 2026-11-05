# Running this project on Google Colab

**Written 2026-11-05.** Colab is the overflow path for decision D9 — local training on the
RTX 2060 is the default, Colab is for when the laptop is needed elsewhere or a phase wants
more GPU than it has.

Nothing in `src/` is Colab-specific. The code reads `N_CMAPSS_DATA` for the data location
and picks its device at runtime, so the same scripts run on a laptop CPU, a local GPU, or a
Colab T4 without edits.

> **Caveat on currency:** Colab's interface and free-tier GPU allocation change over time,
> and this was written from the state of the project rather than from a fresh check of
> Colab's current UI. If a step below does not match what you see, the runtime menu and
> Drive mount are the parts most likely to have moved.

---

## What to put where

| Artefact | Size | Where | Why |
|---|---|---|---|
| Repo | small | `git clone` each session | Always current |
| `ds02_*.parquet` (0.1 Hz) | ~160 MB | Google Drive | Small enough to keep |
| `N-CMAPSS_DS02-006.h5` (1 Hz) | 2.28 GB | Drive, only if needed | Slow; only for the v3 ablation |
| Checkpoints, logs | small | Drive | Must survive disconnection |

**Upload the Parquet, not the HDF5.** After decimation the full development and test sets
are about 160 MB — a one-minute upload. The raw 1 Hz file is 2.28 GB and only the v3
sampling-rate ablation needs it.

---

## Cell 1 — check the GPU

```python
!nvidia-smi
```

Confirms a GPU is attached and which one. If this errors, set the runtime type to GPU
(Runtime → Change runtime type) and run it again.

## Cell 2 — mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

Follow the authorisation prompt. Your files then live under `/content/drive/MyDrive/`.

## Cell 3 — clone the repository

```python
!git clone https://github.com/<your-username>/n-cmapss-ds02-fusion.git /content/repo
%cd /content/repo
```

For a private repo, use a personal access token:

```python
!git clone https://<token>@github.com/<your-username>/n-cmapss-ds02-fusion.git /content/repo
```

Do not paste a token into a notebook you intend to share — Colab saves cell contents.

## Cell 4 — dependencies

```python
!pip install -q pyarrow h5py duckdb
```

Colab ships with torch, numpy, pandas, matplotlib, seaborn and scikit-learn already, and
its torch build matches its own CUDA driver. **Do not reinstall torch** — that usually
replaces a working build with a mismatched one.

Check what you actually got:

```python
import torch, numpy, pandas, pyarrow
print("torch   ", torch.__version__, "| cuda:", torch.cuda.is_available())
print("device  ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("numpy   ", numpy.__version__)
print("pandas  ", pandas.__version__)
print("pyarrow ", pyarrow.__version__)
```

## Cell 5 — point at the data

```python
import os
os.environ["N_CMAPSS_DATA"] = "/content/drive/MyDrive/ncmapss"
```

`src/data_loader.py` checks this variable before falling back to `<repo>/data`. Put the
processed Parquet at `/content/drive/MyDrive/ncmapss/processed/`.

Verify the data loads and matches the paper:

```python
import sys; sys.path.insert(0, "/content/repo")
from src.data_loader import load_processed, unit_summary

print(unit_summary(load_processed("test")).to_string(index=False))
```

Expect units 11, 14, 15 with `t_s` 19 / 36 / 24 and `t_EOL` 59 / 76 / 67. Those are target
paper Table 1 values; if they do not match, the data did not transfer correctly.

## Cell 6 — run the acceptance tests

```python
!cd /content/repo && python -m pytest tests/test_data_loader.py -q
```

Expect `38 passed`.

## Cell 7 — train

```python
!cd /content/repo && python src/train.py --config configs/train_cnn.yaml
```

Written against Phase 4, which does not exist yet. Update this section when `src/train.py`
lands.

## Cell 8 — save results back to Drive

```python
!mkdir -p /content/drive/MyDrive/ncmapss/results
!cp -r /content/repo/results/* /content/drive/MyDrive/ncmapss/results/
```

**Do this after every run.** Colab reclaims the runtime without warning, and anything under
`/content/` that is not on Drive is gone.

---

## Device-agnostic code

Every script picks its device at runtime rather than assuming one:

```python
import torch

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

This will live in `src/utils.py`. The rules that keep code portable:

- Never hard-code `.cuda()` — use `.to(get_device())`
- Never hard-code a Windows or Linux path — derive from `__file__` or an env var
- Save checkpoints with `map_location` in mind, so a GPU-trained model loads on CPU:
  ```python
  torch.load(path, map_location=get_device())
  ```
- Keep batch size in config, not in code — Colab's GPU may have more or less memory than
  the local RTX 2060's 6 GB

---

## Things that will bite you

**Disconnection.** Free Colab reclaims idle runtimes and caps session length. Phase 4 is 30
training runs; do not expect them to finish in one session. Checkpoint every epoch to Drive
and make `train.py` resumable — that is a stated Phase 4 requirement, not just Colab advice.

**GPU allocation varies.** The free tier assigns whatever is available, and heavy use can
drop you to CPU for a while. Check Cell 1 before a long run.

**Drive I/O is slow.** Reading Parquet directly from `/content/drive/` is much slower than
from local disk. Copy to `/content/` first:

```python
!cp -r /content/drive/MyDrive/ncmapss/processed /content/data_local
```

Then point `N_CMAPSS_DATA` at `/content/data_local`. The copy costs seconds and saves
minutes across an epoch.

**Clock drift between runs.** The reproducibility metadata required by
`skills/replication_discipline/SKILL.md` — seed, git commit hash, config hash, wall time,
environment — must be logged on Colab exactly as locally. A Colab run whose provenance is
unrecorded cannot go in the results table.

---

## Related

- `PROJECT_PLAN.md` §1 decision D9 — compute location
- `PROJECT_PLAN.md` §7 — Phase 4 training setup
- `skills/replication_discipline/SKILL.md` — reproducibility metadata per run
