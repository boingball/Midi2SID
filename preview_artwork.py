#!/usr/bin/env python3
"""Render --artwork's dithered output to a PNG, so you can check how a
picture will actually look on the C64 bitmap without building a PRG and
loading it in VICE. Needs Pillow, same as --artwork itself."""
from __future__ import annotations

import argparse
from pathlib import Path

from artwork import C64_PALETTE, CELL_COLS, CELL_ROWS, render_artwork


def decode_to_image(bitmap: bytes, screen: bytes):
    from PIL import Image

    image = Image.new("RGB", (320, 200))
    pixels = image.load()
    for row in range(CELL_ROWS):
        for col in range(CELL_COLS):
            byte = screen[row * CELL_COLS + col]
            ink, paper = C64_PALETTE[byte >> 4], C64_PALETTE[byte & 0xf]
            for line in range(8):
                data = bitmap[row * 320 + col * 8 + line]
                for bit in range(8):
                    on = (data >> (7 - bit)) & 1
                    pixels[col * 8 + bit, row * 8 + line] = ink if on else paper
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="source JPG/PNG")
    parser.add_argument("output", type=Path, help="PNG to write the dithered preview to")
    args = parser.parse_args()
    bitmap, screen = render_artwork(args.image)
    decode_to_image(bitmap, screen).save(args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
