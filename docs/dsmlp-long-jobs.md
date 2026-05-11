# Keeping Long Jobs Alive on DSMLP

DSMLP (DataHub) runs workloads in Kubernetes pods. Two failure modes can kill a long inference run:

1. **SSH disconnect** — closing your terminal or a network drop kills any foreground process.
2. **Pod expiration** — pods are launched with a time limit; when it expires, everything dies.

This document describes the layered approach used to protect multi-hour runs in this project.

---

## Layer 1: tmux (protects against SSH disconnect)

`tmux` creates a terminal session that persists inside the pod even after SSH disconnects. When you reconnect you just re-attach to the session.

```bash
# 1. SSH into the pod as usual.

# 2. Start a named tmux session.
tmux new-session -s phase2_full

# 3. Inside tmux, run the job.
cd ~/151B_SP26_Competition
bash run_phase2_full.sh

# 4. Detach from tmux without killing the job: Ctrl+B then D.
#    The job keeps running inside the pod.

# 5. Later (new SSH connection or after a disconnect), re-attach:
tmux attach -t phase2_full

# 6. List sessions if you forget the name:
tmux list-sessions
```

If tmux is not installed, `screen` is a drop-in alternative:

```bash
screen -S phase2_full
# detach: Ctrl+A then D
# re-attach: screen -r phase2_full
```

---

## Layer 2: Pod time limit (protects against pod expiration)

DSMLP pods expire based on the wall-clock time limit set at launch. The default varies by cluster, but is often 6–8 hours. A full-dataset run (1,126 questions, 8192 max tokens) takes roughly 2–4 hours; a standard pod is usually sufficient, but check before starting.

```bash
# Check remaining lifetime of your current pod.
# activeDeadlineSeconds counts down from pod start, not from now.
kubectl get pod "$(hostname)" -o jsonpath='{.spec.activeDeadlineSeconds}'

# Alternatively, find the pod name and check:
kubectl get pods
kubectl describe pod <pod-name> | grep -i deadline
```

When launching a new pod, request extra hours up front:

```bash
# DSMLP launch with explicit time limit (hours); exact flags vary by cluster config.
# Check your course's launch script or datahub.ucsd.edu documentation.
launch-gpu-pod.sh -g 1 -c 4 -m 32 -t 12   # -t 12 = 12 hours
```

If the pod expires mid-run, see Layer 3 below to recover without re-running from scratch.

---

## Layer 3: Checkpointing (protects against crashes and pod expiration)

`testing_template.py` supports `--checkpoint-every N` and `--resume`:

- `--checkpoint-every 200`: after every 200 questions, score them and flush results to
  `results/{phase}_results.jsonl` and `results/{phase}_errors.jsonl` in append mode.
- `--resume`: on startup, read the existing output files and skip any question IDs already
  present. Inference picks up exactly where it left off.

These flags are baked into `run_phase2_full.sh` (checkpoint every 200; pass `--resume` to the script to activate resume mode).

```bash
# If the job dies at question 600, just re-run with --resume:
bash run_phase2_full.sh --resume
# It will skip questions 1–600 and continue from 601.
```

The final `_summary.json` is only written once, at the very end. Partial runs produce valid
JSONL files but no summary until completion.

---

## Monitoring a Running Job

```bash
# Tail the live log (works from inside or outside the tmux session).
tail -f logs/phase2_full_run.log

# Watch checkpoint progress lines specifically.
grep "Checkpoint saved\|Chunk\|accuracy" logs/phase2_full_run.log

# Check GPU utilization.
nvidia-smi

# Count how many questions have been processed so far.
wc -l results/phase2_full_results.jsonl
```

---

## Full Launch Checklist

Before starting a multi-hour run:

1. **Check pod lifetime.**
   ```bash
   kubectl get pod "$(hostname)" -o jsonpath='{.spec.activeDeadlineSeconds}'
   ```
   If < 14400 (4 hours), launch a new pod with more time.

2. **Start a tmux session.**
   ```bash
   tmux new-session -s phase2_full
   ```

3. **Activate the environment inside tmux.**
   ```bash
   cd ~/151B_SP26_Competition
   # LD_LIBRARY_PATH is set automatically by run_phase2_full.sh
   ```

4. **Launch the job.**
   ```bash
   bash run_phase2_full.sh
   ```

5. **Detach from tmux.** `Ctrl+B D`

6. **Verify from a second terminal.**
   ```bash
   tail -f ~/151B_SP26_Competition/logs/phase2_full_run.log
   ```

---

## What We Did for Each Long Run

### Phase 1 Baseline (2026-05-10)

- **Script:** `run_baseline.py`
- **Command:**
  ```bash
  LD_LIBRARY_PATH=.venv/lib/python3.13/site-packages/torchvision.libs:$LD_LIBRARY_PATH \
    .venv/bin/python run_baseline.py 2>&1 | tee logs/full_baseline_run.log
  ```
- **Job-alive strategy:** None beyond `tee` logging. Run completed without interruption (~4.5 hours).
- **No checkpointing** — if it had crashed, the entire run would have needed to be restarted.

### Phase 2 Full Validation (2026-05-11)

- **Script:** `run_phase2_full.sh` → `testing_template.py --phase phase2_full --full --checkpoint-every 200`
- **Job-alive strategy:** tmux session + checkpoint every 200 questions
- **Log:** `logs/phase2_full_run.log`
- **Resume command (if needed):** `bash run_phase2_full.sh --resume`
