# EqV3 physical direct/clean matched experiment

All six jobs are part of one fixed execution. The first pair is an operational
gate only; completion and numerical checks automatically release the other four
jobs regardless of direction or magnitude of performance differences.

Read LOCK.json for exact settings. Direct and clean both use the full physical
decoder. Clean alone adds the frozen train-fitted composition reference. Network
computation uses bfloat16 in both routes; scalar physical decoding and loss use
float64 to avoid cancellation of large total energies. Every run is exactly40
epochs, no early stop, with native validation per-atom MAE selecting the best.
Post-hoc calibration is identical train-only Ridge on atomic fractions.

Startup gradient gate compares decoded loss and encoded-space loss, including
the outer sigma and beta/sigma conversion. Independently repeated bfloat16
backwards exhibit numerical noise: the gate records that control, requires
identical raw-output gradients and negligible gradient of the loss difference,
and bounds independently accumulated parameter-gradient differences by the
bfloat16 relative machine epsilon. It does not claim bitwise equality of
separately accumulated GPU gradients. Full probe records are in each gate.json.

Track DRIVER_STATUS.json and logs/driver.log. Per-run batch/epoch progress lives
in logs/{route}_{seed}.log; history.csv, epoch0 metrics and val predictions are in
runs/{route}/seed{seed}/. FIRST_PAIR_GATE.json appears only after both first-seed
jobs complete 40 epochs and pass the operational gate.

## Interruption protection

The active execution is owned by the enabled user service
`eqv3-physical-matched.service`, with login lingering enabled. Closing SSH or
Codex therefore does not stop the job; after an abnormal process exit or host
reboot, systemd restarts the driver. `latest.pt` is replaced atomically after
each complete epoch and contains model, optimizer, scheduler, CPU RNG and CUDA
RNG state. `history.csv` and prediction CSVs are also replaced atomically. A
restart resumes at the next epoch only after validating route, seed, lock hash
and the history/checkpoint boundary. A partial epoch is deliberately replayed.

Useful read-only checks:

```bash
systemctl --user status eqv3-physical-matched.service
systemctl --user show eqv3-physical-matched.service -p MainPID -p NRestarts
tail -f logs/driver.log
```

The durable driver automatically runs code/evaluate.py after all six jobs,
writing FINAL_REPORT.md and analysis/. No third backbone or artificial
composition-offset training is scheduled. The same previously inspected
confirmation data are used; no claim of a novel blind cohort is made.

Launch command (do not launch a duplicate while DRIVER_STATUS is active):

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 /home/yxzhang/envs/pet_forme/bin/python experiment17_eqv3_physical_matched_20261001/code/run_all.py
```

An invalid numerical run fails explicitly; no failed or unfavorable seed is
silently omitted. Original experiment14/15/16 checkpoints and references remain
unchanged. MAIN_TEXT_UPDATE.md contains the verified SevenNet main-text material
and withdrawal wording; it is a merge-ready fragment, not a claim that an
unavailable current full manuscript was edited.
