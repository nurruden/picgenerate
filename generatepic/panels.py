"""把用近白间隔拼在一起的结果拆成单张图片。"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

MIN_PANEL = 48
MIN_GUTTER = 2
WHITE = 235


def split_panels(path: Path) -> list[Path]:
    """近白缝隙能切开时返回各张图片；否则只返回原图。"""
    with Image.open(path) as source:
        image = source.convert("RGB")
        boxes = _panel_boxes(image)
        if len(boxes) < 2:
            return [path]
        saved: list[Path] = []
        for index, box in enumerate(boxes, start=1):
            destination = path.with_name(f"{path.stem}-{index}{path.suffix}")
            image.crop(box).save(destination)
            saved.append(destination)
        return saved


def _panel_boxes(image: Image.Image) -> list[tuple[int, int, int, int]]:
    pixels = image.load()
    width, height = image.size
    row_gutters = _gutters(height, lambda index: _row_is_gutter(pixels, index, width))
    bands = _segments(height, row_gutters)
    boxes: list[tuple[int, int, int, int]] = []
    for top, bottom in bands:
        column_gutters = _gutters(
            width,
            lambda index, top=top, bottom=bottom: _column_is_gutter(pixels, index, top, bottom),
        )
        for left, right in _segments(width, column_gutters):
            if right - left >= MIN_PANEL and bottom - top >= MIN_PANEL:
                boxes.append((left, top, right, bottom))
    return boxes


def _row_is_gutter(pixels, y: int, width: int) -> bool:
    return _line_is_white(pixels, ((x, y) for x in range(0, width, 2)))


def _column_is_gutter(pixels, x: int, top: int, bottom: int) -> bool:
    return _line_is_white(pixels, ((x, y) for y in range(top, bottom, 2)))


def _line_is_white(pixels, points) -> bool:
    bright = 0
    total = 0
    for x, y in points:
        red, green, blue = pixels[x, y]
        total += 1
        if min(red, green, blue) >= WHITE:
            bright += 1
    return total > 0 and bright / total >= 0.9


def _gutters(length: int, is_gutter) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index in range(length):
        if is_gutter(index):
            if start is None:
                start = index
            continue
        if start is not None:
            if index - start >= MIN_GUTTER:
                spans.append((start, index))
            start = None
    if start is not None and length - start >= MIN_GUTTER:
        spans.append((start, length))
    return spans


def _segments(length: int, gutters: list[tuple[int, int]]) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    cursor = 0
    for start, end in gutters:
        if start - cursor >= MIN_PANEL:
            segments.append((cursor, start))
        cursor = end
    if length - cursor >= MIN_PANEL:
        segments.append((cursor, length))
    return segments
