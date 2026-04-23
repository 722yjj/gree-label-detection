# AGENTS.md

## Working Mode

- This project is developed directly on the server in the current working checkout.
- If the current checkout contains a project virtual environment such as `.venv`, use that environment first before running `uv` commands, so all installs, checks, and launches stay bound to the current project environment.
- Use Git to manage the repository state on the server: create or switch branches as needed, keep `main` stable, and make changes through normal commit-based workflows.
- Unless the user explicitly asks for a separate local workflow, do not assume code will be edited locally first and then pulled to the server.
- Prefer static checks, unit tests, and dry-run validation in the current server environment; run end-to-end jobs directly on the server when the user asks for them or when the task requires real in-environment verification.
- After code changes, provide the exact command that can be run in the current server checkout to verify the result.

## Canonical Commands

- In this checkout, `.venv/bin/python` is the canonical interpreter once `.venv` exists.
- Desktop app launch command: `.venv/bin/python -m desktop_app.main`
- Root CLI launch command: `.venv/bin/python main.py --template <template_path> --target <target_path>`
- If `.venv` is missing or dependencies need to be refreshed, run `uv sync --extra desktop` in the repo root first, then continue using `.venv/bin/python ...`.
- Avoid mixing `uv run python ...` and `.venv/bin/python ...` in project instructions unless the user explicitly asks for a temporary `uv run` invocation.

## Batch Sample Testing

- The canonical batch command is `.venv/bin/python scripts/batch_run_samples.py`.
- Default discovery scans `samples/pdfs` for template files and `samples/images/produce` for target images.
- Filename matching is code-based: extract the main numeric code from the filename and treat suffixes like `-01` or `_1` as variants of the same label.
- Default pairing mode is `all`: every template variant for a code is compared with every target variant for the same code.
- Batch results should keep only `result.json`, `visualization_diff.jpg`, and the top-level `summary.json`. Intermediate Excel files and debug images should not be preserved in final batch output directories.
- The current default graphic-comparison flow includes composite `image` region splitting; disable it only if you explicitly set `ENABLE_IMAGE_REGION_SPLIT=0`.

## Output Expectations

- Keep result directories stable and easy to sync to the server.
- Prefer paths and commands that work directly in the current server checkout and remain compatible with future Git pulls or fresh clones.
- If batch execution cannot be completed in the current environment because OCR/VLM services are unavailable, leave the script ready and document the recommended server-side command.
