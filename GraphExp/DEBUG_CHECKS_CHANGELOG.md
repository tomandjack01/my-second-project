# Debug Checks Change Log

## Source user instruction
- "??????????"
- "?????????????????????????????????????????????"

## Files changed
- `GraphExp/main_structure_learning.py`
- `GraphExp/models/DDM.py`
- `GraphExp/DEBUG_CHECKS_CHANGELOG.md` (this file)

## What was added
- CLI flag: `--debug_checks`
- One-step check: prints mean `cosine(x_t, x_encoded)` plus t stats (leakage/easiness check).
- One-step check: prints whether temporal encoder grad is `None` (or its norm).
- Fix: allow gradients through `DDM.forward()` layer norm so temporal encoder grads are not accidentally blocked.

- One-step check: prints noise magnitude (abs mean + ratio) and alias/allclose stats for x_t vs x_encoded.
- Change: noise sign preservation is now optional (default off) to avoid inflating cosine similarity.

## How to run
```powershell
python .\GraphExp\main_structure_learning.py --debug_checks --epochs 1 --device cpu
```
