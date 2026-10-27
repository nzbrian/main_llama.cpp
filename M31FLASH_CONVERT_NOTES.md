# M3.1-Flash GGUF conversion - NVFP4 experts + Q8_0 dense

State as of 2026-10-27: conversion COMPLETE. Output is a mixed-quant GGUF:
MoE experts at NVFP4 (4-bit), dense/attention at Q8_0 (8-bit), embed/norms BF16.

## Result
- Output: `/home/crush/512hdd/m31flash_nvfp4.gguf`
- Size: 232.3 GB (249,402,104,448 bytes)
- Tensors: 1359 (171 NVFP4 expert + 420 Q8_0 dense/attn + BF16 embed/norms/scalars)
- vs the earlier Q8_0-only run (456 G): 49% smaller, because the experts
  dropped from 8-bit to 4-bit NVFP4.
- 171 NVFP4 repacks = 57 MoE layers x 3 weights (w1/w2/w3). 3 of the 60
  layers are dense-only (no experts), so 60 - 3 = 57.

## What changed (vs the Q8_0-only approach)
- `conversion/base.py`:
  - mxfp8 branch now SKIPS NVFP4 experts (does not dequant to float).
  - New `_repack_mixed_nvfp4_experts()` repacks each expert directly to GGUF
    NVFP4 (via `_nvfp4_pack`) and merges per (layer, w) into [n_experts, ...].
    No float round-trip, no Q8_0 bloat.
  - Removed the now-dead `dequant_nvfp4` helper.
- `conversion/minimax.py`: in-place MoE merge (unchanged from before; not used
  for the experts, which are repacked directly in base.py).

## Re-run command (if needed)
```
cd /home/crush/src/main_llama.cpp
SNAP=$(ls -d ~/HF/hub/models--MiniMaxAI--MiniMax-M3.1-Flash/snapshots/*/ )
mkdir -p <SPOOL_DISK>/tmp            # MUST exist before launch
TMPDIR=<SPOOL_DISK>/tmp nohup venv/bin/python ./convert_hf_to_gguf.py \
  --outfile <OUTPUT_DISK>/m31flash_nvfp4.gguf \
  --outtype bf16 \
  --fp8-as-q8 \
  --use-temp-file \
  "$SNAP" > /tmp/m31flash_convert.log 2>&1 &
```

## Disk strategy (that worked)
- Spool (`TMPDIR`): `/home/crush/512ssd/tmp` (500G SSD, 586 MB/s)
- Output: `/home/crush/512hdd/m31flash_nvfp4.gguf` (500G HDD, 305 MB/s)
- Spool and output on DIFFERENT disks -> fast final copy (~72 MB/s).
- Spool needs ~233G free; output needs ~233G free. Both disks had 435G free.

## Key gotchas (all hit before)
- `--use-temp-file` CANNOT combine with `--split-max-size`/`--split-max-tensors`.
- `TMPDIR` must point at an EXISTING writable dir on a REAL disk (`/tmp` is tmpfs/RAM here).
- Spool is a `SpooledTemporaryFile`, auto-deleted on exit - no resume.
- RAM stays low (~2-5 GB) with `--use-temp-file` + NVFP4 direct write.
- Dequant/quant phase ~50 min (CPU). Final copy ~35 min on separate disks.

## Verify
- Output file: `ls -la /home/crush/512hdd/m31flash_nvfp4.gguf` (233G)
- Load test with llama-cli (next step).
- Spool cleaned: `rm -rf /home/crush/512ssd/tmp`
