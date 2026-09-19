#!/usr/bin/env python3
"""Convert a JPG/PNG image into a C64 VIC-II hi-res bitmap picture.

Hi-res bitmap mode allows exactly two colours per 8x8 cell (one "ink" bit
colour, one "paper" bit colour) - the same constraint as the Spectrum's
attribute clash MIDI2AY dithers its title card for. This picks the best two
C64-palette colours per cell from the source image, then dithers the whole
image against those fixed per-cell palettes with Floyd-Steinberg error
diffusion, the classic technique behind hand-digitised C64 "hires" photos.

This is the only part of the project that needs a third-party dependency
(Pillow, for JPEG/PNG decoding and resizing) - importing it is deferred to
render_artwork() so the rest of midi2sid stays dependency-free unless
--artwork is actually used.
"""
from __future__ import annotations

from pathlib import Path

BITMAP_WIDTH = 320
BITMAP_HEIGHT = 200
CELL_COLS = BITMAP_WIDTH // 8
CELL_ROWS = BITMAP_HEIGHT // 8
BITMAP_SIZE = BITMAP_WIDTH * BITMAP_HEIGHT // 8
SCREEN_SIZE = CELL_ROWS * CELL_COLS

# The "pepto" C64 palette: a widely used, well-measured approximation of the
# 6569 PAL VIC-II's actual output colours, in VIC colour-register order.
C64_PALETTE = (
    (0x00, 0x00, 0x00), (0xff, 0xff, 0xff), (0x68, 0x37, 0x2b), (0x70, 0xa4, 0xb2),
    (0x6f, 0x3d, 0x86), (0x58, 0x8d, 0x43), (0x35, 0x28, 0x79), (0xb8, 0xc7, 0x6f),
    (0x6f, 0x4f, 0x25), (0x43, 0x39, 0x00), (0x9a, 0x67, 0x59), (0x44, 0x44, 0x44),
    (0x6c, 0x6c, 0x6c), (0x9a, 0xd2, 0x84), (0x6c, 0x5e, 0xb5), (0x95, 0x95, 0x95),
)


def cell_index(row: int, col: int) -> int:
    return row * CELL_COLS + col


TWO_COLOUR_GAIN_THRESHOLD = 0.20


def _cell_two_colours(pixels: list[tuple[int, int, int]]) -> tuple[int, int]:
    """Pick the two C64 palette entries that best represent this cell.

    A continuous 2-means split followed by snapping each cluster's average
    colour to the nearest palette entry can land on a poor pair: an average
    computed in RGB space isn't guaranteed to sit near a good real palette
    colour. 16 colours is small enough (120 distinct pairs) to instead
    brute-force the exact best single colour and the exact best distinct
    pair - whichever minimises the summed squared error against every pixel
    in the cell, each pixel measured against its closer colour of the pair.

    Always taking the best distinct pair sounds strictly better, but it
    isn't: Floyd-Steinberg only dithers a cell when its ink and paper
    differ, so handing every near-flat cell (a gradient sky, a plain
    background) two genuinely different colours - even when they win by a
    tiny margin - makes it dither too, turning a clean flat area into
    speckle. Real detail (a hard edge, a facial feature) needs the second
    colour; a gentle gradient does not. Requiring the pair to cut error by
    TWO_COLOUR_GAIN_THRESHOLD keeps two colours for the former and one flat
    colour for the latter.
    """
    if not pixels:
        return 0, 0
    distances_by_colour = [
        [sum((p - c) ** 2 for p, c in zip(pixel, colour)) for pixel in pixels]
        for colour in C64_PALETTE
    ]
    best_single, best_single_error = 0, None
    for i, distances in enumerate(distances_by_colour):
        error = sum(distances)
        if best_single_error is None or error < best_single_error:
            best_single, best_single_error = i, error
    best_pair, best_pair_error = (best_single, best_single), best_single_error
    for i in range(len(C64_PALETTE)):
        for j in range(i + 1, len(C64_PALETTE)):
            error = sum(map(min, distances_by_colour[i], distances_by_colour[j]))
            if error < best_pair_error:
                best_pair, best_pair_error = (i, j), error
    if best_single_error == 0 or best_pair_error >= best_single_error * (1 - TWO_COLOUR_GAIN_THRESHOLD):
        return best_single, best_single
    return best_pair


def _cover_resize(image, width: int, height: int):
    from PIL import Image

    src_w, src_h = image.size
    scale = max(width / src_w, height / src_h)
    new_w, new_h = max(1, round(src_w * scale)), max(1, round(src_h * scale))
    image = image.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - width) // 2
    top = (new_h - height) // 2
    return image.crop((left, top, left + width, top + height))


def render_artwork(path: str | Path) -> tuple[bytes, bytes]:
    """Convert an image file into (bitmap[8000 bytes], screen[1024 bytes])."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "--artwork needs Pillow to decode/resize the image "
            "(pip install Pillow); the rest of midi2sid has no dependencies."
        ) from exc

    image = _cover_resize(Image.open(path).convert("RGB"), BITMAP_WIDTH, BITMAP_HEIGHT)
    source = list(image.getdata())  # row-major RGB tuples, BITMAP_WIDTH*BITMAP_HEIGHT

    def pixel_at(x: int, y: int) -> tuple[int, int, int]:
        return source[y * BITMAP_WIDTH + x]

    cell_colours: list[tuple[int, int]] = []
    for row in range(CELL_ROWS):
        for col in range(CELL_COLS):
            block = [
                pixel_at(col * 8 + x, row * 8 + y)
                for y in range(8) for x in range(8)
            ]
            cell_colours.append(_cell_two_colours(block))

    screen = bytearray(SCREEN_SIZE + 24)  # pad to a clean 1024 for a 4-page copy
    for index, (ink, paper) in enumerate(cell_colours):
        screen[index] = (ink << 4) | paper

    # Floyd-Steinberg dithering against each cell's own fixed two colours,
    # carried in raster order across the whole image (not per cell), which
    # is what lets dithering hide the block boundaries between cells.
    error = [[0.0, 0.0, 0.0] for _ in range(BITMAP_WIDTH)]
    next_error = [[0.0, 0.0, 0.0] for _ in range(BITMAP_WIDTH)]
    bitmap = bytearray(BITMAP_SIZE)
    for y in range(BITMAP_HEIGHT):
        row = y // 8
        line = y % 8
        for x in range(BITMAP_WIDTH):
            col = x // 8
            ink_index, paper_index = cell_colours[cell_index(row, col)]
            ink_rgb, paper_rgb = C64_PALETTE[ink_index], C64_PALETTE[paper_index]
            original = pixel_at(x, y)
            wanted = tuple(o + e for o, e in zip(original, error[x]))
            dist_ink = sum((a - b) ** 2 for a, b in zip(wanted, ink_rgb))
            dist_paper = sum((a - b) ** 2 for a, b in zip(wanted, paper_rgb))
            use_ink = dist_ink <= dist_paper
            chosen = ink_rgb if use_ink else paper_rgb
            if use_ink:
                byte_index = row * 320 + col * 8 + line
                bitmap[byte_index] |= 0x80 >> (x % 8)
            diff = [w - c for w, c in zip(wanted, chosen)]
            if x + 1 < BITMAP_WIDTH:
                for channel in range(3):
                    error[x + 1][channel] += diff[channel] * 7 / 16
            if x > 0:
                for channel in range(3):
                    next_error[x - 1][channel] += diff[channel] * 3 / 16
            for channel in range(3):
                next_error[x][channel] += diff[channel] * 5 / 16
            if x + 1 < BITMAP_WIDTH:
                for channel in range(3):
                    next_error[x + 1][channel] += diff[channel] * 1 / 16
        error, next_error = next_error, [[0.0, 0.0, 0.0] for _ in range(BITMAP_WIDTH)]

    return bytes(bitmap), bytes(screen)
