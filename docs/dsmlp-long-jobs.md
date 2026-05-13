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

2. **Record the node hostname.** Copy this somewhere you can reach from outside the pod
   (a local notes file, the run log header, or a quick message to yourself).
   ```bash
   hostname   # e.g. dsmlp-gpu-13.ucsd.edu
   echo "Node: $(hostname)  Started: $(date)" >> logs/<run>.log
   ```
   Without this, you cannot SSH back in to check on the job if you close your terminal.

3. **Start a tmux session.**
   ```bash
   tmux new-session -s phase2_full
   ```

4. **Activate the environment inside tmux.**
   ```bash
   cd ~/151B_SP26_Competition
   # LD_LIBRARY_PATH is set automatically by run_phase2_full.sh
   ```

5. **Launch the job.**
   ```bash
   bash run_phase2_full.sh
   ```

6. **Detach from tmux.** `Ctrl+B D`

7. **Verify from a second terminal.**
   ```bash
   tail -f ~/151B_SP26_Competition/logs/phase2_full_run.log
   ```

---

## Alternative: Claude Code Background Jobs + Loop Monitor

When running Claude Code inside the JupyterHub container (not via a real SSH terminal), `tmux`
fails with "open terminal failed: not a terminal" because Claude Code's shell is not a TTY.
`ssh <hostname>` also fails ("Connection refused") because the JupyterHub pod does not expose
port 22. In this environment, use Claude Code's own background-task and loop facilities instead.

### Layer 1 replacement: background task

Run the job via the Bash tool with `run_in_background: true`. Claude Code tracks the process,
writes its output to a temp file, and notifies you on completion. The process is decoupled from
the active conversation turn — it survives between turns as long as the session stays open.

```
# In a Claude Code prompt:
"Run bash scripts/run_phase3_stage1_resume.sh in the background"
```

This gives you a task ID (e.g. `btlfrsbxg`) you can use to stop or tail the job.

### Layer 2 replacement: /loop for keepalive + monitoring

JupyterHub can cull idle user servers. A `/loop` running every few minutes keeps the session
active and gives you live progress updates:

```
/loop Every 4 minutes, tail the last 20 lines of logs/phase3_stage1_resume.log and report
current phase, questions done, accuracy, and whether the background task is still running.
Stop when both summary files exist or the job errors out.
```

The loop calls `ScheduleWakeup` at ~240 s intervals, re-enters itself each turn, and stops
when the termination condition is met.

### Limitations vs. tmux

| | tmux | Claude Code background + /loop |
|---|---|---|
| Survives SSH disconnect | Yes | Yes (process is backgrounded) |
| Survives session close / browser tab close | Yes | **No** — session must stay open |
| Survives pod expiration | No (both die) | No (both die) |
| Live monitoring | Manual `tail -f` | Automatic via /loop |
| Keepalive against idle culling | Not needed (tmux activity counts) | Yes — /loop turns count as activity |

If you are on a machine where tmux works (real SSH terminal), prefer tmux. Use this strategy
only when running entirely through the Claude Code JupyterHub session.

### Phase 3 Stage 1 Resume (2026-05-13)

- **Script:** `scripts/run_phase3_stage1_resume.sh` (temp07 resume from Q100, temp09 fresh)
- **Background task ID:** `btlfrsbxg`
- **Job-alive strategy:** Claude Code background task + `/loop` every 4 min
- **Log:** `logs/phase3_stage1_resume.log`
- **Node:** `dsmlp-jupyter-doterocaldwell`
- **Resume command (if needed):** re-run `bash scripts/run_phase3_stage1_resume.sh` as a
  background task — the `--resume` flag in the script will skip completed checkpoints

### Phase 3 Stage 2 + Final Run (2026-05-13)

- **Script:** `scripts/run_phase3_pipeline.sh` (Stage 1 skipped — all summary files existed; pipeline resumed at Stage 2)
- **tmux session:** `p3_pipeline` (detached: `tmux new-session -d -s p3_pipeline`)
- **Job-alive strategy:** detached tmux session — works from JupyterHub because `-d` avoids the TTY requirement; the session persists inside the pod even without an interactive terminal
- **Log:** `logs/phase3_pipeline.log`
- **Node:** `dsmlp-jupyter-doterocaldwell`
- **Launched:** 2026-05-13 18:35 UTC
- **Monitoring:**
  ```bash
  tmux attach -t p3_pipeline          # reattach (if you have a real SSH terminal)
  tail -f logs/phase3_pipeline.log    # watch progress from any shell
  grep "accuracy\|done\|complete\|Starting" logs/phase3_pipeline.log
  wc -l results/phase3_mv3_results.jsonl   # checkpoint progress
  ```
- **Resume command (if needed):**
  ```bash
  tmux new-session -d -s p3_pipeline
  tmux send-keys -t p3_pipeline "bash scripts/run_phase3_pipeline.sh" Enter
  ```
  All completed runs skip (summary file check); interrupted runs resume from last checkpoint (`--resume` flag in script).

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

### Phase 3 Temperature Sweep (2026-05-11 – 2026-05-12)

- **Script:** `scripts/run_phase3_pipeline.sh` → `testing_template.py` with varying `--temperature`
- **Configs run:** temp=0.3 ✓, temp=0.5 ✓, temp=0.6 ✓, temp=0.7 ✗ (assumed dead — node not recorded)
- **Job-alive strategy:** tmux + checkpoint-based resume (same as Phase 2)
- **Log:** `logs/phase3_pipeline.log`
- **Lesson:** The temp=0.7 run started on 2026-05-12 ~06:47, last log entry at 06:58 (prompt 30/50
  of chunk 1). Node hostname was not recorded so the job could not be checked remotely after
  disconnect. Assumed dead; re-run with node hostname logged at start.
- **Resume command:** `bash scripts/run_phase3_pipeline.sh --resume` (skips completed temp configs)
