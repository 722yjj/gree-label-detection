# AGENTS.md

## Working Mode

- This project is usually edited locally, then pushed to the remote repository, and finally pulled and executed on the server.
- Unless the user explicitly asks for local end-to-end execution, prefer static checks, unit tests, and dry-run validation locally.
- If the user explicitly authorizes direct server-side or in-environment execution, you may run end-to-end tests or batch jobs directly in the available runtime (for this project, typically `conda run -n ocr ...`).
- After code changes, provide the exact server-side command the user can run after pulling the latest code.

## Batch Sample Testing

- The batch entrypoint is `python scripts/batch_run_samples.py`.
- Default discovery scans `samples/pdfs` for template files and `samples/images/produce` for target images.
- Filename matching is code-based: extract the main numeric code from the filename and treat suffixes like `-01` or `_1` as variants of the same label.
- Default pairing mode is `all`: every template variant for a code is compared with every target variant for the same code.
- Batch results should keep only `result.json`, `visualization_diff.jpg`, and the top-level `summary.json`. Intermediate Excel files and debug images should not be preserved in final batch output directories.
- The current default graphic-comparison flow includes composite `image` region splitting; no extra env flag is required unless you want to disable it explicitly with `ENABLE_IMAGE_REGION_SPLIT=0`.

## Output Expectations

- Keep result directories stable and easy to sync to the server.
- Prefer paths and commands that work both locally and after the repository is pulled on the server.
- If batch execution cannot be completed locally because OCR/VLM services are unavailable, leave the script ready for server execution and document the recommended command.
