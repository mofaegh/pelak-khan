# Changelog

All notable changes to Pelak-Khan are documented in this file.

## [0.5.0] - 2026-08-08

### Added

- Unified image-recognition API
- Reusable detector, OCR, validator, tracking, and storage runtime components
- Responsive Persian RTL web interface
- Image upload and drag-and-drop recognition
- Live webcam recognition
- Video upload and asynchronous processing workflow
- Video sampling controls and progress reporting
- Temporal OCR voting
- Sharpest-frame selection
- Live/video duplicate suppression and temporal tracking
- Plate region/province lookup for supported codes
- SQLite schema migration support
- Detection history and search
- Delete individual detections
- Delete plate history
- Clear-database workflow with explicit confirmation
- SQLite backup endpoint/workflow
- GitHub developer/repository branding in the UI
- Windows portable launcher
- Bundled Python runtime packaging strategy for AI dependencies
- Portable ZIP and SHA-256 release generation
- Automated tests for validator, region lookup, tracking, and database deletion

### Changed

- Refactored recognition logic out of one-off scripts into reusable runtime services
- Expanded FastAPI backend for image, live, video, history, database, and media workflows
- Improved Iranian plate rendering in RTL layouts
- Added Persian digit presentation while preserving internal normalized values
- Improved live recognition with temporal stabilization
- Reworked Windows packaging after PyInstaller dependency-graph limitations with the full AI stack
- Disabled Uvicorn's default logging configuration in portable mode to avoid runtime formatter failures

### Packaging

The Windows v0.5.0 portable release uses a lightweight `Pelak-Khan.exe` launcher plus a bundled Python runtime and dependencies. The application is distributed as a ZIP and requires no separate Python installation on the target machine.

### Known limitations

- Production-grade high-speed ALPR performance has not yet been fully benchmarked across representative operational camera setups.
- Plate-format coverage is not yet complete for every Iranian plate class.
- Public redistribution of trained model weights remains subject to verification of underlying dataset/model licensing terms.

