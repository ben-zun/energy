# Reference-allocation final experiments

This directory implements the frozen experiment order supplied on 2026-10-04.
Code is authored and tested locally, then copied to `/home/yxzhang/energy/experiment_reference_allocation_final_20261004` for data-bound evaluation and GPU work.

The authoritative protocol is the user attachment. `scripts/freeze_protocol.py` copies it byte-for-byte into `PROTOCOL.md` and records its SHA256 before any experiment runs.

Execution gates:

1. freeze protocol and configuration;
2. run local unit and static tests;
3. inventory and hash server inputs without modifying them;
4. finish EqV3 evaluation;
5. run required-vs-induced analyses;
6. construct and test synthetic labels;
7. run one smoke seed for program correctness only;
8. run all three fixed seeds regardless of result direction;
9. audit force definitions and summarize MP→r²SCAN;
10. run the final numerical audit.

No acceptance item is marked passed until its corresponding check has run.
