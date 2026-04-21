# AGENTS.md

## Working Mode

- This project is developed directly on the server in the current working checkout.
- Use Git to manage the repository state on the server: create or switch branches as needed, keep `main` stable, and make changes through normal commit-based workflows.
- Unless the user explicitly asks for a separate local workflow, do not assume code will be edited locally first and then pulled to the server.
- Prefer static checks, unit tests, and dry-run validation in the current server environment; run end-to-end jobs directly on the server when the user asks for them or when the task requires real in-environment verification.
- After code changes, provide the exact command that can be run in the current server checkout to verify the result.

## Batch Sample Testing

- The batch entrypoint is `python scripts/batch_run_samples.py`.
- Default discovery scans `samples/pdfs` for template files and `samples/images/produce` for target images.
- Filename matching is code-based: extract the main numeric code from the filename and treat suffixes like `-01` or `_1` as variants of the same label.
- Default pairing mode is `all`: every template variant for a code is compared with every target variant for the same code.
- Batch results should keep only `result.json`, `visualization_diff.jpg`, and the top-level `summary.json`. Intermediate Excel files and debug images should not be preserved in final batch output directories.

## Output Expectations

- Keep result directories stable and easy to sync to the server.
- Prefer paths and commands that work directly in the current server checkout and remain compatible with future Git pulls or fresh clones.
- If batch execution cannot be completed in the current environment because OCR/VLM services are unavailable, leave the script ready and document the recommended server-side command.
