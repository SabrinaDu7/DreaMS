# Nibi launch-babysit-fix loop

A runbook for launching a DreaMS pre-training run on the Nibi (Digital
Research Alliance of Canada) cluster, watching it until it reaches a target
amount of training progress, and — if it crashes first — diagnosing why,
proposing a fix, and relaunching once the fix is approved. Written so a
fresh Claude session with no memory of prior conversations can execute this
loop by reading this file alone.

## Prerequisites

- **SSH access**: `ssh nibi` must work non-interactively. This repo has
  `nibi_connect.py` at the repo root, which establishes a persistent SSH
  ControlMaster socket (Nibi requires MFA; the socket avoids repeating it).
  Check first with `ssh -O check nibi`. If it says "Master running", you're
  set. If not, the human needs to run `python3 nibi_connect.py` themselves
  and approve a Duo push on their phone — this is not something a Claude
  session can do unattended.
- **Repo location on Nibi**: `~/experiments/DreaMS` (persistent clone).
- **Dataset**: `$SCRATCH/datasets/gems/GeMS_A10.hdf5` (~14G, gzip-compressed
  on disk — decompresses much larger in RAM, see "Known failure modes"
  below).
- **wandb**: entity `sabrina-du-mcgill-university`, project `dreams`. Auth
  key lives at `~/experiments/DreaMS/wandb_api.txt` on Nibi (gitignored,
  not in the repo). Locally, `wandb login` must have been run at least once
  so `~/.netrc` has cached credentials — the babysit script queries the
  wandb API directly (not over SSH) to read training progress.
- **The training script**: `dreams/training/pre_train.sh`. Takes one
  optional positional arg, `max_epochs` (default 3000):
  `sbatch dreams/training/pre_train.sh <max_epochs>`.

## Step 1 — Preflight checks (run before every submission)

```bash
ssh nibi '
  bash -n ~/experiments/DreaMS/dreams/training/pre_train.sh && echo OK
  ls -lh "$SCRATCH/datasets/gems/GeMS_A10.hdf5"
  ls -la ~/experiments/DreaMS/wandb_api.txt
  ls -ld ~/scratch/dreams_runs || mkdir -p ~/scratch/dreams_runs
  sq
'
```

`sq`/`squeue` should show no unexpected lingering jobs. `~/scratch/dreams_runs`
**must exist before submission** — SLURM opens the `--output`/`--error`
files (both point there) before the script body runs, so a missing
directory fails the job immediately.

If you (the agent) have local uncommitted or unpushed changes to
`pre_train.sh` relative to what's on Nibi, reconcile them explicitly first
— `git diff` / `git log --oneline -3 -- dreams/training/pre_train.sh` on
both sides. Don't assume the two checkouts match; they have diverged before
(local fixes not yet pushed, or hand-edits made directly on Nibi during a
previous debugging session).

## Step 2 — Determine the target `global_step`

The babysit script watches Lightning's `trainer/global_step` metric (logged
to wandb), not epochs directly, because the SLURM stdout log is an
unreliable progress signal (see "Known failure modes"). Steps-per-epoch
depends on dataset size ÷ `batch_size` and isn't hardcoded anywhere —
**ask the user** for the target step count (e.g. "1 epoch is ~95k global
steps") rather than guessing from file size. If genuinely unknown, you can
back it out empirically after the first few minutes of a run: dataset
spectra count ÷ `--batch_size` (256 as of this writing) ≈ steps/epoch.

## Step 3 — Submit

```bash
ssh nibi 'cd ~/experiments/DreaMS && sbatch dreams/training/pre_train.sh <max_epochs>'
```

Note the returned job ID. Confirm it queued:

```bash
ssh nibi sq
```

## Step 4 — Babysit

Use `dreams/training/nibi_babysit.sh <job_id> <target_global_step>
[wandb_entity/project]`. Run it under the Monitor tool with `persistent:
true` (queue wait time is unbounded, and a full epoch can take hours) —
**not** a plain foreground `Bash` call, and not a manual sleep-poll loop.

```
Monitor({
  command: "cd ~/experiments/DreaMS && ./dreams/training/nibi_babysit.sh <job_id> <target_step>",
  description: "Babysit Nibi job <job_id> to <target_step> global steps",
  persistent: true,
  timeout_ms: 3600000
})
```

The script emits one line per notable event — job state transitions,
finding the wandb run id, 10k-step progress milestones, and two terminal
conditions:

- `TARGET_REACHED: ...` — **success**. The script exits 0. Report back to
  the user and stop watching. **Do not cancel the job** — it keeps running
  toward its full `max_epochs` on its own; babysitting only covers the
  smoke-test window the user cares about.
- `JOB_GONE: ...` — the job left the SLURM queue before reaching the
  target. Go to Step 5.
- `NO_WANDB_RUN: ...` — **check `sq`/`sacct` before assuming a crash.** If
  the job is still `R` (running), this can be a false alarm: the discovery
  window in `nibi_babysit.sh` can still be too tight for a slow run (rsync
  of the dataset, a fresh `uv sync`, and the retention-order-pairs indexing
  step have been observed to legitimately take 35+ minutes combined). If
  `sq` still shows it running, query wandb directly for the newest run in
  the project instead of re-running discovery from scratch, confirm
  `global_step` is actually advancing, and if so just resume Phase 3
  monitoring against that run id — don't treat it as Step 5.

Do not poll manually while this runs — the notification arrives as a new
message when something happens. If asked for status mid-wait, say so;
don't fabricate a progress guess.

## Step 5 — On crash: diagnose, propose, wait for approval, relaunch

1. **Diagnose.** SSH in and gather evidence — don't guess:
   ```bash
   ssh nibi "sacct -j <job_id> --format=JobID,State,ExitCode,Elapsed,MaxRSS -j <job_id>"
   ssh nibi "tail -100 ~/scratch/dreams_runs/DreaMS_pre-training_result_<job_id>.out"
   ssh nibi "tail -100 ~/scratch/dreams_runs/DreaMS_pre-training_result_<job_id>.err"
   ```
   Check `sacct`'s `State`/`ExitCode` first — it tells you *what kind* of
   failure this was (OOM, walltime, non-zero exit, node failure, `CANCELLED`)
   before you go spelunking in the log. Cross-check against "Known failure
   modes" below; this loop has hit the same handful of issues more than
   once.
2. **Propose a fix.** Write up: what broke, why, and the exact diff you'd
   apply to `pre_train.sh` (or elsewhere in the repo). Present it to the
   user.
3. **Wait for explicit approval.** Do not edit `pre_train.sh` or anything
   else, locally or on Nibi, until the user says go. This is a hard rule,
   not a default-yes.
4. **Apply, once approved.** Make the same edit on whichever checkout is
   the source of truth for the next run (check — local and Nibi have
   drifted before; you may need to push/pull or edit directly over SSH).
5. **Relaunch** — back to Step 1 (preflight) and Step 3 (submit), then
   resume babysitting at Step 4 with a fresh job ID.

## Known failure modes (check these before assuming something new)

- **`AttributeError: _ARRAY_API not found` on `import rdkit`** — a
  cluster-loaded Python/SciPy module leaks a `PYTHONPATH` that shadows the
  venv's numpy with a different build, breaking rdkit's compiled C
  extension ABI. Fixed by `unset PYTHONPATH` near the top of
  `pre_train.sh`, before `uv sync`. This is already in the script — if it
  recurs, check the fix wasn't accidentally reverted, and check
  `echo $PYTHONPATH` / `module list` on Nibi to see what's leaking in.
- **`slurmstepd: error: Detected 1 oom_kill event`** — this is a **system
  RAM** OOM (SLURM cgroup killer), not GPU VRAM — a real CUDA OOM instead
  shows a Python `torch.cuda.OutOfMemoryError` traceback. Root cause:
  `MaskedSpectraDataset.__init__` (`dreams/utils/data.py`) unconditionally
  loads the *entire* hdf5 dataset into RAM (`f[k][:]`, no lazy/mmap path).
  The dataset file is gzip-compressed on disk, so its size on disk (~14G)
  is **not** its size in RAM once decompressed — expect substantially more.
  `--num_workers_data` > 0 multiplies this further (each DataLoader worker
  process). Levers: raise `--mem` in the SBATCH header, and/or lower
  `--num_workers_data`. As of this writing these have been tuned
  iteratively (8 → 4 workers, `--mem=64G` kept as a deliberate test) — check
  current values in the script rather than assuming.
- **pandas/matchms/numpy version drift** — `pyproject.toml` previously had
  unbounded lower-bound-only version constraints (e.g. `pandas>=2.2.1`)
  that silently resolved to breaking major versions (pandas 3.0 changed
  string column dtypes; a `matchms` release renamed
  `ModifiedCosine`→`ModifiedCosineGreedy`). Both are already pinned/fixed.
  If a similar "works locally, breaks after a fresh `uv lock`" bug shows up
  again, suspect the same pattern first: check `uv.lock` for what actually
  resolved vs. what's pinned in `pyproject.toml`.
- **MIG GPU slices are not extra compute.** If tempted to request many
  `nvidia_h100_80gb_hbm3_1g.10gb` slices to "parallelize" — don't. MIG
  splits *one* physical GPU into fractions; it doesn't add GPUs. The
  numbers SLURM prints in a "specify a GPU type" error (e.g. `h100:232`,
  `nvidia_h100_80gb_hbm3_1g.10gb:128`) are the **cluster-wide total pool
  size** of each type, not a per-job quantity to request.
- **GPU idle / near-0% utilization** — check `--num_workers_data` isn't 0
  (synchronous single-threaded loading starves the GPU) and that
  `--cpus-per-task` in the SBATCH header actually covers it. Also note:
  `HMMA`/`FP16 Pipeline` utilization near 0% is *expected* under
  `--train_precision 32` (fp32 doesn't route through tensor cores) — that
  specific metric being low isn't itself evidence of a data-loading
  problem; check overall SM/GPU-Util instead.

## Reference

- Training launch script: `dreams/training/pre_train.sh`
- Babysit/monitor script: `dreams/training/nibi_babysit.sh`
- SSH connection helper: `nibi_connect.py` (repo root)
