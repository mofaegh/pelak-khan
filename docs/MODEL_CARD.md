# Pelak-Khan Model Card

## Overview

Pelak-Khan v0.5.0 uses a two-stage ALPR pipeline:

1. license-plate detection
2. cropped-plate OCR followed by format validation and post-processing

The application also uses temporal tracking/voting for live and video workflows to improve stability across consecutive frames.

## Detector

Runtime model path:

```text
models/runtime/detector_v1.pt
```

Model family: Ultralytics YOLO, single `plate` class, 640-pixel inference/training size.

Held-out detector test results used for the v0.5.0 project benchmark:

| Metric | Result |
| --- | ---: |
| Precision | 97.44% |
| Recall | 100% |
| mAP@50 | 99.33% |
| mAP@50:95 | 90.73% |

These numbers describe the evaluated held-out dataset and are not a guarantee of production performance.

## OCR

Runtime model path:

```text
models/runtime/ocr_v1.pt
```

The OCR model is a CRNN-style recognizer using a CNN feature extractor, bidirectional LSTM sequence modeling, and CTC decoding. Input plate crops are normalized to grayscale at 128x32.

Held-out OCR results used for the v0.5.0 project benchmark:

| Metric | Result |
| --- | ---: |
| Exact plate accuracy | 88.41% |
| Character Error Rate (CER) | 2.56% |

## Post-processing

The v0.5.0 validator is strongest on standard Iranian private vehicle plates. The internal normalized representation follows the private-plate pattern used by the project, with display formatting applied separately for the Persian UI.

Live/video processing can combine multiple observations using temporal voting, duplicate suppression, and sharp-frame selection.

## Intended use

Pelak-Khan is intended for research, prototyping, local automation, and evaluation of Iranian license-plate recognition workflows. It should be benchmarked on representative cameras and operating conditions before operational deployment.

## Known limitations

Performance may degrade with motion blur, low resolution, glare, severe perspective, occlusion, poor lighting, dirty plates, nonstandard fonts, compression artifacts, or plate layouts outside the currently supported rules.

Special/public/taxi/government/diplomatic/disabled/motorcycle formats are not fully covered by the v0.5.0 validator.

High-speed real-world traffic performance has not been established by the held-out dataset metrics alone.

## Privacy and responsible deployment

License plates can be personal or sensitive operational data depending on jurisdiction and context. Deployers are responsible for access controls, retention policies, lawful use, signage/notice where required, and protection of stored media and databases.

## Licensing note

Model and dataset licensing is separate from the repository source-code license. See `THIRD_PARTY_NOTICES.md` and `docs/DATASETS.md` before redistributing trained weights.
