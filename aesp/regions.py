"""Validate submitted xyxy boxes in original-image pixels; never convert units."""
import math


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{name} must be a finite number")
    return value


def validate_dimensions(width, height):
    for name, value in (("width", width), ("height", height)):
        if _finite(value, name) <= 0:
            raise ValueError(f"{name} must be positive")
    return width, height


def validate_box(value, width, height):
    """Require [xmin, ymin, xmax, ymax]; preserve values without clipping."""
    validate_dimensions(width, height)
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("box must be a list of four numbers: [xmin, ymin, xmax, ymax]")
    box = [_finite(x, "box coordinate") for x in value]
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("box must satisfy 0 <= xmin < xmax <= image_width and "
                         "0 <= ymin < ymax <= image_height in original-image pixels")
    return box


def validate_prediction_regions(prediction, width, height):
    """Read evidence_boxes as original-image pixels; ignore other fields."""
    validate_dimensions(width, height)
    if 'evidence_boxes' not in prediction:
        raise ValueError("Prediction missing required field: evidence_boxes")
    boxes = prediction['evidence_boxes']
    if not isinstance(boxes, list):
        raise ValueError("evidence_boxes must be a list of xyxy boxes; use [] for no boxes")
    checked = []
    for index, box in enumerate(boxes):
        try:
            checked.append(validate_box(box, width, height))
        except ValueError as exc:
            raise ValueError(f"evidence_boxes[{index}]: {exc}") from exc
    return checked
