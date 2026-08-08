# Pelak-Khan

**Pelak-Khan** is a local-first Iranian license plate recognition (ALPR) application for Windows and the web. It combines plate detection, OCR, validation, temporal tracking, SQLite history, image/video processing, and live-camera recognition behind a responsive Persian RTL interface.

> Version: **v0.5.0**  
> Developer: **[@mofaegh](https://github.com/mofaegh)**

## Highlights

- Iranian license plate detection
- OCR for Persian plate characters
- Plate-format validation
- Region / province lookup for supported plate codes
- Image recognition
- Live webcam recognition
- Video processing with progress reporting
- Temporal OCR voting and sharpest-frame selection
- Duplicate suppression / tracking for repeated live detections
- SQLite-backed detection history
- Search and filtering
- Delete individual detections or plate history
- Database backup
- Responsive Persian RTL web UI
- Local-only FastAPI backend
- Windows portable build: extract and run `Pelak-Khan.exe`

## Portable Windows Release

The recommended way to use Pelak-Khan on Windows is the portable release.

1. Download `Pelak-Khan-Portable-Windows-x64-v0.5.0.zip` from GitHub Releases.
2. Extract the ZIP to a writable folder.
3. Run `Pelak-Khan.exe`.
4. The local backend starts automatically and the default browser opens.

No separate Python, virtual environment, pip, FastAPI, PyTorch, or Ultralytics installation is required on the target machine.

Portable user data is kept inside the application folder under `data/`, so the complete folder can be moved between systems together with its history and saved media.

## Application Architecture

```text
Browser UI
   |
   v
FastAPI backend (localhost)
   |
   +-- Detector
   +-- OCR
   +-- Plate validator
   +-- Region lookup
   +-- Temporal tracker / OCR voting
   +-- SQLite storage
   +-- Image / video / live-camera workflows
```

The desktop executable is a lightweight launcher. The Windows portable package contains a bundled Python runtime and runtime dependencies next to the application rather than freezing the complete AI stack into one large executable.

## Development Setup

### Requirements

- Windows 10/11 or a compatible Python environment
- Python 3.10

Create and activate a virtual environment, then install the project and runtime requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements-runtime.txt
```

### Runtime models

Pelak-Khan expects the runtime model files at:

```text
models/runtime/detector_v1.pt
models/runtime/ocr_v1.pt
```

Model weights are intentionally excluded from normal Git history.

### Run the web application

```powershell
uvicorn apps.backend.main:app --host 127.0.0.1 --port 8000
```

Then open:

```text
http://127.0.0.1:8000/
```

API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## Main API

### Health check

```http
GET /health
```

### Image recognition

```http
POST /api/recognize/image
```

The recognition pipeline performs upload handling, plate detection, crop extraction, OCR, validation, media storage, SQLite ingestion, and JSON response generation.

## Windows Portable Build

Install build-only dependencies:

```powershell
python -m pip install -r requirements-build.txt
```

Build the portable release:

```powershell
powershell -ExecutionPolicy Bypass -File .\build\windows\build_portable.ps1 -Version 0.5.0
```

Release artifacts are generated under:

```text
release/
```

Expected files:

```text
Pelak-Khan-Portable-Windows-x64-v0.5.0.zip
Pelak-Khan-Portable-Windows-x64-v0.5.0.sha256.txt
```

## Tests

Run the automated tests with:

```powershell
python -m pytest -q
```

The current test suite covers core behavior including plate validation, regional metadata, temporal tracking/voting, database migration and deletion workflows.

## Repository Policy

The following are intentionally excluded from normal Git commits:

- virtual environments
- local databases and runtime user data
- trained model weights
- inference/training artifacts
- generated portable builds
- release ZIP files
- temporary development patches

Portable binaries belong in **GitHub Releases**, not in normal repository history.

## Current Scope and Limitations

Pelak-Khan v0.5.0 is a practical beta release, not a guarantee of production-grade recognition under every camera, lighting, viewing-angle, motion-blur, or vehicle-speed condition.

The current validator primarily targets supported Iranian private-plate formats. Additional plate classes such as public, taxi, government, diplomatic, disabled, and motorcycle layouts may require dedicated rule sets and additional training/validation work.

High-speed recognition should be benchmarked against representative cameras, shutter settings, frame rates, vehicle speeds, viewing angles, and environmental conditions before operational deployment.

## Security / Privacy

The standard desktop configuration binds the backend to `127.0.0.1`, keeping the application local to the machine. Avoid exposing the FastAPI service directly to untrusted networks without adding appropriate authentication, TLS, upload controls, and deployment hardening.

## Model and Dataset Licensing

Source-code licensing does not automatically grant permission to redistribute third-party datasets, dataset-derived artifacts, pretrained components, or trained model weights. Before distributing model files publicly, verify the licenses and redistribution terms of every dataset and dependency used to produce them.

## License

The source code in this repository is released under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. See [LICENSE](LICENSE).

Third-party libraries and model/data assets remain subject to their own licenses and terms.

## Author

Developed by **[@mofaegh](https://github.com/mofaegh)**.

Repository: **https://github.com/mofaegh/pelak-khan**
