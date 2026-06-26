# Dataset Closed-Loop Validation

This document records the current automated closed loop between the external
`dataset` project and `gree-label-detection`.

## Goal

Use the `dataset` project to create synthetic abnormal label samples, run the
same desktop detection service path used by `gree-label-detection`, and verify
whether the final detected problem boxes overlap the generated GT boxes.

The manual photo workflow is still separate: print a generated abnormal label,
open the desktop app, take a real photo, and inspect the app result.

## Repositories

- Dataset generator: `/home/jnu/projects/dataset`
- Detection project: `/home/jnu/projects/gree-label-detection`

## One-Command Flow

Run from `gree-label-detection`:

```bash
.venv/bin/python scripts/run_dataset_closed_loop.py \
  --source-dir /home/jnu/projects/dataset/inputs/pdfs \
  --text-only-count 2 \
  --seed 20260625 \
  --generate-retries 5
```

For a fixed output directory:

```bash
.venv/bin/python scripts/run_dataset_closed_loop.py \
  --source-dir /home/jnu/projects/dataset/inputs/pdfs \
  --output-root /home/jnu/projects/gree-label-detection/results/dataset_closed_loop/actual_20260626-200730 \
  --text-only-count 2 \
  --seed 20260625 \
  --generate-retries 5 \
  --overwrite
```

For one source PDF, use `--source-pdf /path/to/source.pdf` instead of
`--source-dir`.

## Steps

1. `generate_dataset`
   Calls `dataset/scripts/build_dataset.py` and creates abnormal label PDFs and
   preview images. The current generator focuses on text-region mutations and
   records mutation bboxes in `text_debug.mutations`.

2. `convert_dataset`
   Calls `scripts/prepare_dataset_project_eval_dataset.py` and converts dataset
   project output into `visual_manifest.json`, target PNGs, source templates,
   sample PDFs, and GT preview images.

3. `ensure_vlm`
   Calls `scripts/start_vllm.sh` before detection when using the default
   `traditional_full_image_diff` desktop pipeline. The runner also reads
   `scripts/start_vllm.sh --print-env` and passes the desktop launcher VLM
   environment into the evaluation process.

   This matters because directly calling `DetectionService` without the desktop
   launcher layer can miss `LLM_PROVIDER=vllm`, `OPENAI_COMPATIBLE_API_BASE`,
   model names, and the local vLLM startup.

   If vLLM is already managed manually, pass `--skip-vlm-start`.

4. `evaluate_dataset`
   Calls `scripts/evaluate_synthetic_dataset.py`, which builds desktop
   `DetectionJobRequest` objects and runs `DetectionService` with
   `DESKTOP_DETECTION_PIPELINE=traditional_full_image_diff`.

## Validation Rule

The current synthetic text GT is the PDF text element box affected by the
mutation, not only the changed character glyph.

The desktop traditional flow maps template PDF text-line regions into the
target image, attaches diff candidates to those regions, then sends candidates
to VLM. Therefore the final result box should be close to a predefined text
region. Validation checks final predicted boxes from the desktop result against
GT boxes:

- A case passes when each expected GT region is matched by at least one final
  predicted box.
- Text GT uses overlap/span coverage because a predicted text-line box can be
  larger than the exact mutated glyph span.
- Extra final boxes are recorded as `extra`, but the main failure condition is
  currently missing expected GT boxes.
- Per-case detection time is recorded as `detection_duration_seconds`.

## Output Layout

Given an output root such as:

```text
results/dataset_closed_loop/actual_20260626-200730
```

Important files are:

```text
closed_loop_summary.json
evaluation/summary.json
evaluation_dataset/visual_manifest.json
evaluation_dataset/images/<sample_id>.png
evaluation_dataset/pdfs/<sample_id>.pdf
evaluation/cases/<sample_id>/evaluation_result.json
evaluation/cases/<sample_id>/final_result.json
evaluation/cases/<sample_id>/visualization_diff.jpg
evaluation/cases/<sample_id>/ground_truth_boxes.png
evaluation/cases/<sample_id>/dataset_preview.png
evaluation/representative_failure/comparison.jpg
```

For visual inspection:

- `visualization_diff.jpg` is the final detected-box visualization.
- `ground_truth_boxes.png` is the GT-box visualization.
- `representative_failure/comparison.jpg` stitches template, target, and diff
  visualization for the first failed case.

## Latest Actual Run

Command:

```bash
.venv/bin/python scripts/run_dataset_closed_loop.py \
  --source-dir /home/jnu/projects/dataset/inputs/pdfs \
  --output-root /home/jnu/projects/gree-label-detection/results/dataset_closed_loop/actual_20260626-200730 \
  --text-only-count 2 \
  --seed 20260625 \
  --generate-retries 5 \
  --overwrite
```

Result:

```text
total=10
passed=8
failed=2
pass_rate=0.8
detection_duration_seconds.average=23.171
detection_duration_seconds.median=24.175
box_stats.expected=15
box_stats.predicted=17
box_stats.matched=13
box_stats.missing=2
box_stats.extra=4
```

Failed cases:

```text
600001076226_sample_0001: expected=2 predicted=3 matched=1 missing=1 extra=2
600004083205_sample_0002: expected=2 predicted=1 matched=1 missing=1 extra=0
```

Primary summaries:

```text
/home/jnu/projects/gree-label-detection/results/dataset_closed_loop/actual_20260626-200730/closed_loop_summary.json
/home/jnu/projects/gree-label-detection/results/dataset_closed_loop/actual_20260626-200730/evaluation/summary.json
```
