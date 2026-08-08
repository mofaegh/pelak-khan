# Contributing to Pelak-Khan

Thanks for your interest in contributing to Pelak-Khan.

## Development environment

Pelak-Khan currently targets Python 3.10 and Windows for the portable desktop release.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements-runtime.txt
python -m pip install -r requirements-build.txt
```

Runtime model files are expected at:

```text
models/runtime/detector_v1.pt
models/runtime/ocr_v1.pt
```

Model weights, local databases, datasets, generated artifacts, and portable build outputs must not be committed to normal Git history.

## Running tests

```powershell
python -m pytest -q
```

A change should not be merged if it breaks the existing test suite.

## Pull requests

Keep pull requests focused on one feature or fix. Include a concise description of what changed, why it changed, and how it was tested. For UI changes, screenshots are strongly recommended. For recognition changes, include representative test cases and avoid reporting benchmark improvements without a reproducible evaluation procedure.

## Code and data contributions

Do not contribute datasets, images, videos, model weights, or other third-party assets unless redistribution rights are clear. When a contribution depends on external data or models, document the source and applicable license/terms.

## Security issues

Please follow `SECURITY.md` for security-related reports. Do not publish credentials, personal data, or exploitable security details in a public issue.
