#!/usr/bin/env python3
"""Smart MIDI-to-C64 SID converter."""
from __future__ import annotations

import argparse
from pathlib import Path

from build_prg import build_prg, encode_events, pack_lzss
from sid_midi import compile_sid_frames


def convert(
    midi: Path, output: Path, title: str | None, video: str,
    drums: str, filter_mode: str, feel: str, artwork: Path | None = None,
) -> None:
    frames = compile_sid_frames(
        midi, video=video, drums=drums, filter_mode=filter_mode, feel=feel,
    )
    artwork_bytes = None
    if artwork is not None:
        from artwork import render_artwork
        artwork_bytes = render_artwork(artwork)
    build_prg(frames, output, title or midi.stem, video=video, artwork=artwork_bytes)
    event_size = len(encode_events(frames))
    packed_size = len(pack_lzss(encode_events(frames)))
    print(
        f"wrote {output} ({len(frames)} {video.upper()} frames, "
        f"{event_size} event bytes, {packed_size} LZSS-packed, "
        f"drums={drums}, filter={filter_mode}, feel={feel}"
        + (f", artwork={artwork.name}" if artwork is not None else "")
        + ")"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a Standard MIDI file into a self-running Commodore 64 SID PRG."
    )
    parser.add_argument("midi", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--title", help="title shown on the generated C64 screen")
    parser.add_argument("--video", choices=("pal", "ntsc"), default="pal")
    parser.add_argument("--drums", choices=("off", "smart"), default="smart")
    parser.add_argument("--filter", choices=("off", "auto"), default="off",
                        help="shared SID filter; off is the clean 6581-safe default")
    parser.add_argument("--feel", choices=("tight", "expressive"), default="tight",
                        help="tight gives precise chip attacks; expressive keeps slower GM envelopes")
    parser.add_argument("--artwork", type=Path,
                        help="JPG/PNG background image, dithered to a C64 hi-res bitmap "
                             "(needs Pillow: pip install Pillow)")
    args = parser.parse_args()
    convert(
        args.midi, args.output, args.title, args.video, args.drums, args.filter, args.feel,
        artwork=args.artwork,
    )


if __name__ == "__main__":
    main()
