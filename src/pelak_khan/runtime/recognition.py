from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

from pelak_khan.ocr.crnn import CRNN, ctc_greedy_decode, load_checkpoint
from pelak_khan.postprocessing.plate_validator import (
    display_plate,
    edge_suspect,
    expand_box,
    validate_standard_plate,
)


class RecognitionService:
    """Reusable Pelak-Khan image recognition runtime."""

    def __init__(
        self,
        detector_path: Path,
        ocr_path: Path,
        device: str = "auto",
        det_imgsz: int = 640,
        pad_frac: float = 0.04,
        edge_margin_frac: float = 0.01,
    ) -> None:
        self.detector_path = Path(detector_path)
        self.ocr_path = Path(ocr_path)
        self.det_imgsz = det_imgsz
        self.pad_frac = pad_frac
        self.edge_margin_frac = edge_margin_frac

        if not self.detector_path.exists():
            raise FileNotFoundError(f"Detector model not found: {self.detector_path}")
        if not self.ocr_path.exists():
            raise FileNotFoundError(f"OCR model not found: {self.ocr_path}")

        self.device = torch.device(
            "cuda:0"
            if device == "auto" and torch.cuda.is_available()
            else ("cpu" if device == "auto" else device)
        )

        self.detector = YOLO(str(self.detector_path))

        checkpoint = load_checkpoint(self.ocr_path, self.device)
        self.charset = list(checkpoint["charset"])
        self.allowed_letters = {ch for ch in self.charset if not ch.isdigit()}
        self.idx_to_char = {index + 1: ch for index, ch in enumerate(self.charset)}
        self.blank_idx = int(checkpoint.get("blank_idx", 0))

        self.ocr_model = CRNN(
            num_classes=int(checkpoint["num_classes"]),
            hidden_size=int(checkpoint["hidden_size"]),
        ).to(self.device)
        self.ocr_model.load_state_dict(checkpoint["model_state"])
        self.ocr_model.eval()

        self.ocr_transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize(
                    (int(checkpoint["image_h"]), int(checkpoint["image_w"])),
                    antialias=True,
                ),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def recognize_frame(
        self,
        frame,
        *,
        source_image: str = "",
        crops_dir: Path | None = None,
        crop_path_prefix: str = "frame",
        annotated_path: Path | None = None,
        annotated_relative_path: str = "",
        det_conf: float = 0.25,
        allow_edge: bool = False,
        save_crops: bool = True,
        save_annotated: bool = True,
    ) -> dict[str, Any]:
        """Recognize a BGR OpenCV frame.

        For ordinary image uploads, save_crops/save_annotated remain True.
        Live-camera requests can set both False to avoid writing every sampled
        frame to disk; the backend persists only deduplicated accepted events.
        """
        if save_crops:
            if crops_dir is None:
                raise ValueError("crops_dir is required when save_crops=True")
            crops_dir.mkdir(parents=True, exist_ok=True)

        if save_annotated:
            if annotated_path is None:
                raise ValueError("annotated_path is required when save_annotated=True")
            annotated_path.parent.mkdir(parents=True, exist_ok=True)

        height, width = frame.shape[:2]
        annotated = frame.copy() if save_annotated else None

        prediction = self.detector.predict(
            source=frame,
            imgsz=self.det_imgsz,
            conf=det_conf,
            device=str(self.device),
            verbose=False,
        )[0]

        records: list[dict[str, Any]] = []
        boxes = prediction.boxes

        if boxes is not None:
            for plate_index, box in enumerate(boxes, start=1):
                x1, y1, x2, y2 = box.xyxy[0].detach().cpu().tolist()
                detection_confidence = float(box.conf[0].detach().cpu())

                is_edge_suspect = edge_suspect(
                    x1,
                    y1,
                    x2,
                    y2,
                    width,
                    height,
                    self.edge_margin_frac,
                )

                cx1, cy1, cx2, cy2 = expand_box(
                    x1,
                    y1,
                    x2,
                    y2,
                    width,
                    height,
                    self.pad_frac,
                )
                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size == 0:
                    continue

                crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                sharpness = float(cv2.Laplacian(crop_gray, cv2.CV_64F).var())
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                tensor = (
                    self.ocr_transform(Image.fromarray(crop_rgb))
                    .unsqueeze(0)
                    .to(self.device)
                )

                with torch.inference_mode():
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=torch.float16,
                        enabled=self.device.type == "cuda",
                    ):
                        logits = self.ocr_model(tensor)

                text = ctc_greedy_decode(
                    logits,
                    self.idx_to_char,
                    self.blank_idx,
                )[0]

                format_valid, reasons = validate_standard_plate(
                    text,
                    self.allowed_letters,
                )
                accepted = format_valid and (allow_edge or not is_edge_suspect)
                if is_edge_suspect and not allow_edge:
                    reasons.append("touches_image_edge")

                status = "ACCEPTED" if accepted else "REJECTED"
                color = (0, 200, 0) if accepted else (0, 0, 220)

                crop_relative = ""
                if save_crops and crops_dir is not None:
                    crop_name = f"{crop_path_prefix}_plate_{plate_index:02d}.jpg"
                    crop_file = crops_dir / crop_name
                    cv2.imwrite(str(crop_file), crop)
                    crop_relative = f"crops/{crop_name}"

                if annotated is not None:
                    cv2.rectangle(
                        annotated,
                        (int(round(x1)), int(round(y1))),
                        (int(round(x2)), int(round(y2))),
                        color,
                        2,
                    )
                    cv2.putText(
                        annotated,
                        f"{status} det={detection_confidence:.3f}",
                        (int(round(x1)), max(20, int(round(y1)) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

                records.append(
                    {
                        "source_image": source_image,
                        "plate_index": plate_index,
                        "raw_text": text,
                        "display_text": display_plate(text),
                        "det_confidence": detection_confidence,
                        "sharpness": sharpness,
                        "format_valid": format_valid,
                        "edge_suspect": is_edge_suspect,
                        "accepted": accepted,
                        "status": status,
                        "reject_reasons": reasons,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "crop_path": crop_relative,
                    }
                )

        if annotated is not None and annotated_path is not None:
            cv2.imwrite(str(annotated_path), annotated)

        return {
            "detections": len(records),
            "accepted": sum(1 for item in records if item["accepted"]),
            "rejected": sum(1 for item in records if not item["accepted"]),
            "annotated_path": annotated_relative_path if save_annotated else "",
            "results": records,
        }
