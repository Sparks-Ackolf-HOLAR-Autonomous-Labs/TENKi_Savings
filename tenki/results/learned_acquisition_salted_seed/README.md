# Superseded: experiment 27 run with a salted training seed

This folder holds the original experiment 27 run (September 2026). Its
cross-entropy training was seeded with Python's `hash()`, which is salted
per interpreter, so the run cannot be reproduced. Experiment 27b
(`learned_acquisition_long/`) evaluated the per-fold weights stored here,
so the folder is kept for that provenance.

The reproducible rerun (seed 42, `zlib.crc32` seeding) is in
`../learned_acquisition/`; four further training seeds and their summary
are in `../learned_acquisition_seeds/`.
