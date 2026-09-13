#!/usr/bin/env python3
"""Build a self-running C64 PRG containing a 50/60 Hz SID event player."""
from __future__ import annotations

from pathlib import Path

LOAD_ADDRESS = 0x0801
ENTRY_ADDRESS = 0x080d
EVENT_DELAY = 0xfd
EVENT_FRAME = 0xfe
EVENT_END = 0xff
MAX_END_ADDRESS = 0xcfff


class Assembler:
    """Tiny label-aware assembler for the handful of 6502 opcodes we need."""

    def __init__(self, origin: int):
        self.origin = origin
        self.code = bytearray()
        self.labels: dict[str, int] = {}
        self.fixups: list[tuple[int, str, str]] = []

    @property
    def pc(self) -> int:
        return self.origin + len(self.code)

    def label(self, name: str) -> None:
        self.labels[name] = self.pc

    def byte(self, *values: int) -> None:
        self.code.extend(value & 0xff for value in values)

    def imm(self, opcode: int, value: int) -> None:
        self.byte(opcode, value)

    def zp(self, opcode: int, address: int) -> None:
        self.byte(opcode, address)

    def absolute(self, opcode: int, address: int | str) -> None:
        self.byte(opcode)
        if isinstance(address, str):
            self.fixups.append((len(self.code), address, "absolute"))
            self.byte(0, 0)
        else:
            self.byte(address, address >> 8)

    def branch(self, opcode: int, label: str) -> None:
        self.byte(opcode)
        self.fixups.append((len(self.code), label, "relative"))
        self.byte(0)

    def resolve(self) -> bytes:
        for offset, label, kind in self.fixups:
            if label not in self.labels:
                raise ValueError(f"undefined 6502 label: {label}")
            target = self.labels[label]
            if kind == "absolute":
                self.code[offset] = target & 0xff
                self.code[offset + 1] = target >> 8
            else:
                source_after_operand = self.origin + offset + 1
                displacement = target - source_after_operand
                if not -128 <= displacement <= 127:
                    raise ValueError(f"branch to {label} is out of range")
                self.code[offset] = displacement & 0xff
        return bytes(self.code)


def encode_events(frames: list[bytes]) -> bytes:
    """Delta-compress complete SID snapshots into register writes and delays."""
    previous = bytes(25)
    output = bytearray()
    delay = 0

    def flush_delay() -> None:
        nonlocal delay
        while delay:
            count = min(255, delay)
            output.extend((EVENT_DELAY, count))
            delay -= count

    for frame in frames:
        if len(frame) != 25:
            raise ValueError("every SID frame must contain exactly 25 registers")
        changes = [(register, value) for register, value in enumerate(frame) if value != previous[register]]
        if not changes:
            delay += 1
            continue
        flush_delay()
        # Switching directly between gated waveforms produces a hard edge,
        # particularly when voice three changes from a noise drum back to a
        # bass waveform. Release GATE before programming the new voice.
        for control in (4, 11, 18):
            old, new = previous[control], frame[control]
            if old & 1 and new & 1 and (old & 0xf0) != (new & 0xf0):
                output.extend((control, old & 0xfe))
        for register, value in changes:
            output.extend((register, value))
        output.append(EVENT_FRAME)
        previous = frame
    flush_delay()
    # Silence all gates and master volume before looping.
    output.extend((4, 0, 11, 0, 18, 0, 24, 0, EVENT_FRAME, EVENT_END))
    return bytes(output)


def decode_events(events: bytes, frame_limit: int = 1_000_000) -> list[bytes]:
    """Reference decoder used by tests to verify the player stream."""
    registers = bytearray(25)
    frames: list[bytes] = []
    position = 0
    while position < len(events) and len(frames) < frame_limit:
        command = events[position]
        position += 1
        if command == EVENT_END:
            return frames
        if command == EVENT_FRAME:
            frames.append(bytes(registers))
        elif command == EVENT_DELAY:
            count = events[position]
            position += 1
            frames.extend([bytes(registers)] * count)
        elif command < 25:
            registers[command] = events[position]
            position += 1
        else:
            raise ValueError(f"invalid SID event command ${command:02x}")
    raise ValueError("SID event stream is missing its end marker")


def pack_lzss(data: bytes) -> bytes:
    """Pack bytes with an intentionally tiny 256-byte-window LZSS format.

    Each control byte describes eight tokens, least-significant bit first.
    A set bit is one literal byte. A clear bit is a two-byte back-reference:
    distance-minus-one followed by length-minus-three. The 6502 player decodes
    this incrementally into a single page at $0200; no full unpacked copy is
    ever needed in C64 memory.
    """
    output = bytearray()
    position = 0
    while position < len(data):
        control_at = len(output)
        output.append(0)
        control = 0
        for bit in range(8):
            if position >= len(data):
                break
            best_length = best_distance = 0
            for distance in range(1, min(256, position) + 1):
                if data[position - distance] != data[position]:
                    continue
                length = 1
                while (
                    length < 258
                    and position + length < len(data)
                    and data[position + length] == data[position - distance + length]
                ):
                    length += 1
                if length >= 3 and length > best_length:
                    best_length, best_distance = length, distance
            if best_length >= 3:
                output.extend((best_distance - 1, best_length - 3))
                position += best_length
            else:
                control |= 1 << bit
                output.append(data[position])
                position += 1
        output[control_at] = control
    return bytes(output)


def unpack_lzss(data: bytes) -> bytes:
    """Reference implementation of the C64 streaming decompressor."""
    output = bytearray()
    position = 0
    while position < len(data):
        control = data[position]
        position += 1
        for bit in range(8):
            if position >= len(data):
                break
            if control & (1 << bit):
                output.append(data[position])
                position += 1
            else:
                distance = data[position] + 1
                length = data[position + 1] + 3
                position += 2
                for _ in range(length):
                    output.append(output[-distance])
    return bytes(output)


def screen_code(character: str) -> int:
    character = character.upper()
    if "A" <= character <= "Z":
        return ord(character) - 64
    if "0" <= character <= "9":
        return ord(character)
    return {" ": 32, "-": 45, ".": 46, "/": 47, ":": 58}.get(character, 32)


def _basic_stub() -> bytes:
    # 10 SYS2061, followed by the zero next-line pointer. Machine code starts
    # at $080d (2061), immediately after this standard BASIC v2 launcher.
    return bytes((0x0b, 0x08, 0x0a, 0x00, 0x9e, 0x32, 0x30, 0x36, 0x31, 0x00, 0x00, 0x00))


def _player(title: str, packed_events: bytes, video: str) -> bytes:
    assembler = Assembler(ENTRY_ADDRESS)
    zp_colour, zp_frame, zp_mode = 0xf2, 0xf3, 0xf4
    zp_register, zp_ring = 0xf5, 0xf6
    zp_flags, zp_bits = 0xf7, 0xf8
    zp_copy_len, zp_copy_pos = 0xf9, 0xfa
    zp_lo, zp_hi, zp_wait = 0xfb, 0xfc, 0xfd

    # Initial machine setup and a clean generated text-mode title screen.
    assembler.byte(0x78)                         # SEI
    assembler.imm(0xa9, 0x35); assembler.zp(0x85, 0x01)
    assembler.imm(0xa9, 0); assembler.absolute(0x8d, 0xd020); assembler.absolute(0x8d, 0xd021)
    assembler.imm(0xa2, 0)
    assembler.imm(0xa9, 32)
    assembler.label("clear_screen")
    for address in (0x0400, 0x0500, 0x0600, 0x0700):
        assembler.absolute(0x9d, address)
    assembler.byte(0xe8); assembler.branch(0xd0, "clear_screen")
    assembler.imm(0xa2, 0); assembler.imm(0xa9, 1)
    assembler.label("clear_colour")
    for address in (0xd800, 0xd900, 0xda00, 0xdb00):
        assembler.absolute(0x9d, address)
    assembler.byte(0xe8); assembler.branch(0xd0, "clear_colour")

    for label, address in (
        ("logo", 0x0488), ("title", 0x0501),
        ("voice1", 0x05e2), ("voice2", 0x05f0), ("voice3", 0x05fe),
        ("keys1", 0x06d1), ("keys2", 0x06f9),
    ):
        assembler.imm(0xa2, 0)
        assembler.label(f"copy_{label}")
        assembler.absolute(0xbd, label)
        assembler.branch(0xf0, f"done_{label}")
        assembler.absolute(0x9d, address)
        assembler.byte(0xe8)
        assembler.branch(0xd0, f"copy_{label}")
        assembler.label(f"done_{label}")

    assembler.imm(0xa2, 24); assembler.imm(0xa9, 0)
    assembler.label("clear_sid")
    assembler.absolute(0x9d, 0xd400)
    assembler.byte(0xca); assembler.branch(0x10, "clear_sid")
    assembler.imm(0xa9, 15); assembler.absolute(0x8d, 0xd418)
    assembler.imm(0xa9, 0)
    for address in (zp_colour, zp_frame, zp_ring, zp_flags, zp_bits, zp_copy_len, zp_copy_pos, zp_wait):
        assembler.zp(0x85, address)
    assembler.imm(0xa9, 1); assembler.zp(0x85, zp_mode)
    # CIA 1 port A selects keyboard rows; port B reads columns.
    assembler.imm(0xa9, 0xff); assembler.absolute(0x8d, 0xdc02)
    assembler.imm(0xa9, 0x00); assembler.absolute(0x8d, 0xdc03)
    # Event pointer immediates are patched after labels resolve.
    data_lo_at = len(assembler.code) + 1
    assembler.imm(0xa9, 0); assembler.zp(0x85, zp_lo)
    data_hi_at = len(assembler.code) + 1
    assembler.imm(0xa9, 0); assembler.zp(0x85, zp_hi)

    raster = 250 if video == "pal" else 235
    assembler.label("wait_leave")
    assembler.absolute(0xad, 0xd012); assembler.imm(0xc9, raster); assembler.branch(0xf0, "wait_leave")
    assembler.label("wait_raster")
    assembler.absolute(0xad, 0xd012); assembler.imm(0xc9, raster); assembler.branch(0xd0, "wait_raster")
    assembler.absolute(0x20, "update")
    assembler.absolute(0x20, "visual")
    assembler.absolute(0x4c, "wait_leave")

    assembler.label("update")
    assembler.zp(0xa5, zp_wait); assembler.branch(0xf0, "parse")
    assembler.zp(0xc6, zp_wait); assembler.byte(0x60)
    assembler.label("parse")
    assembler.absolute(0x20, "getbyte")
    assembler.imm(0xc9, EVENT_END); assembler.branch(0xf0, "restart")
    assembler.imm(0xc9, EVENT_FRAME); assembler.branch(0xf0, "return")
    assembler.imm(0xc9, EVENT_DELAY); assembler.branch(0xf0, "set_delay")
    # The streaming decompressor uses X for its ring buffer, so retain the
    # target SID register in zero page while fetching the value byte.
    assembler.zp(0x85, zp_register)
    assembler.absolute(0x20, "getbyte")
    assembler.zp(0xa6, zp_register)
    assembler.absolute(0x9d, 0xd400)
    assembler.absolute(0x4c, "parse")
    assembler.label("set_delay")
    assembler.absolute(0x20, "getbyte"); assembler.byte(0x38); assembler.imm(0xe9, 1)
    assembler.zp(0x85, zp_wait)
    assembler.label("return"); assembler.byte(0x60)
    assembler.label("restart")
    restart_lo_at = len(assembler.code) + 1
    assembler.imm(0xa9, 0); assembler.zp(0x85, zp_lo)
    restart_hi_at = len(assembler.code) + 1
    assembler.imm(0xa9, 0); assembler.zp(0x85, zp_hi)
    assembler.imm(0xa9, 0)
    for address in (zp_ring, zp_flags, zp_bits, zp_copy_len, zp_copy_pos, zp_wait):
        assembler.zp(0x85, address)
    assembler.byte(0x60)

    assembler.label("getbyte")
    # Continue a pending LZSS back-reference one byte at a time.
    assembler.zp(0xa5, zp_copy_len); assembler.branch(0xf0, "new_token")
    assembler.zp(0xa6, zp_copy_pos)                 # LDX zp
    assembler.absolute(0xbd, 0x0200)               # LDA $0200,X
    assembler.zp(0xe6, zp_copy_pos); assembler.zp(0xc6, zp_copy_len)
    assembler.absolute(0x4c, "emit_byte")

    assembler.label("new_token")
    assembler.zp(0xa5, zp_bits); assembler.branch(0xd0, "have_flags")
    assembler.absolute(0x20, "getpacked"); assembler.zp(0x85, zp_flags)
    assembler.imm(0xa9, 8); assembler.zp(0x85, zp_bits)
    assembler.label("have_flags")
    assembler.zp(0x46, zp_flags)                   # LSR flags, literal bit -> carry
    assembler.zp(0xc6, zp_bits)
    assembler.branch(0x90, "back_reference")       # BCC
    assembler.absolute(0x20, "getpacked")
    assembler.absolute(0x4c, "emit_byte")

    assembler.label("back_reference")
    assembler.absolute(0x20, "getpacked")          # distance - 1
    assembler.byte(0x18); assembler.imm(0x69, 1)    # CLC / ADC #1
    assembler.zp(0x85, zp_copy_pos)
    assembler.zp(0xa5, zp_ring); assembler.byte(0x38)
    assembler.zp(0xe5, zp_copy_pos)                 # SBC zp
    assembler.zp(0x85, zp_copy_pos)
    assembler.absolute(0x20, "getpacked")          # length - 3
    assembler.byte(0x18); assembler.imm(0x69, 3)
    assembler.zp(0x85, zp_copy_len)
    assembler.absolute(0x4c, "getbyte")

    assembler.label("emit_byte")
    assembler.zp(0xa6, zp_ring)
    assembler.absolute(0x9d, 0x0200)
    assembler.zp(0xe6, zp_ring)
    assembler.byte(0x60)

    assembler.label("getpacked")
    assembler.imm(0xa0, 0); assembler.byte(0xb1, zp_lo)
    assembler.zp(0xe6, zp_lo); assembler.branch(0xd0, "gotbyte")
    assembler.zp(0xe6, zp_hi)
    assembler.label("gotbyte"); assembler.byte(0x60)

    assembler.label("scan_keys")
    # C64 matrix: 1/2 on row 7, 3/4 on row 1, and 5 on row 2.
    assembler.imm(0xa9, 0x7f); assembler.absolute(0x8d, 0xdc00)
    assembler.absolute(0xad, 0xdc01); assembler.imm(0x29, 0x01); assembler.branch(0xd0, "key2")
    assembler.imm(0xa9, 1); assembler.zp(0x85, zp_mode); assembler.absolute(0x4c, "keys_done")
    assembler.label("key2")
    assembler.absolute(0xad, 0xdc01); assembler.imm(0x29, 0x08); assembler.branch(0xd0, "row1")
    assembler.imm(0xa9, 2); assembler.zp(0x85, zp_mode); assembler.absolute(0x4c, "keys_done")
    assembler.label("row1")
    assembler.imm(0xa9, 0xfd); assembler.absolute(0x8d, 0xdc00)
    assembler.absolute(0xad, 0xdc01); assembler.imm(0x29, 0x01); assembler.branch(0xd0, "key4")
    assembler.imm(0xa9, 3); assembler.zp(0x85, zp_mode); assembler.absolute(0x4c, "keys_done")
    assembler.label("key4")
    assembler.absolute(0xad, 0xdc01); assembler.imm(0x29, 0x08); assembler.branch(0xd0, "row2")
    assembler.imm(0xa9, 4); assembler.zp(0x85, zp_mode); assembler.absolute(0x4c, "keys_done")
    assembler.label("row2")
    assembler.imm(0xa9, 0xfb); assembler.absolute(0x8d, 0xdc00)
    assembler.absolute(0xad, 0xdc01); assembler.imm(0x29, 0x01); assembler.branch(0xd0, "keys_done")
    assembler.imm(0xa9, 5); assembler.zp(0x85, zp_mode)
    assembler.label("keys_done")
    assembler.imm(0xa9, 0xff); assembler.absolute(0x8d, 0xdc00); assembler.byte(0x60)

    assembler.label("draw_lights")
    for control, cell in ((0xd404, 0x0635), (0xd40b, 0x0643), (0xd412, 0x0651)):
        assembler.absolute(0xad, control); assembler.imm(0x29, 1)
        assembler.branch(0xf0, f"off_{cell:04x}")
        assembler.imm(0xa9, 0xa0); assembler.branch(0xd0, f"draw_{cell:04x}")
        assembler.label(f"off_{cell:04x}"); assembler.imm(0xa9, 32)
        assembler.label(f"draw_{cell:04x}"); assembler.absolute(0x8d, cell)
    assembler.byte(0x60)

    assembler.label("blank_lights")
    assembler.imm(0xa9, 32)
    for cell in (0x0635, 0x0643, 0x0651):
        assembler.absolute(0x8d, cell)
    assembler.byte(0x60)

    assembler.label("slow_colour")
    assembler.zp(0xa5, zp_frame); assembler.imm(0x29, 0x0f); assembler.branch(0xd0, "slow_done")
    assembler.zp(0xe6, zp_colour); assembler.zp(0xa5, zp_colour); assembler.imm(0x29, 0x0f)
    assembler.absolute(0x8d, 0xd020)
    assembler.label("slow_done"); assembler.byte(0x60)

    assembler.label("visual")
    assembler.absolute(0x20, "scan_keys"); assembler.zp(0xe6, zp_frame)
    assembler.zp(0xa5, zp_mode); assembler.imm(0xc9, 1); assembler.branch(0xf0, "mode_safe")
    assembler.imm(0xc9, 2); assembler.branch(0xf0, "mode_lights")
    assembler.imm(0xc9, 3); assembler.branch(0xf0, "mode_border")
    assembler.imm(0xc9, 4); assembler.branch(0xf0, "mode_both")
    # Mode 5 remains deliberately slow: colour changes at roughly 3 Hz PAL.
    assembler.absolute(0x20, "draw_lights"); assembler.absolute(0x20, "slow_colour")
    assembler.zp(0xa5, zp_colour); assembler.imm(0x29, 7); assembler.absolute(0x8d, 0xd021); assembler.byte(0x60)
    assembler.label("mode_safe")
    assembler.imm(0xa9, 0); assembler.absolute(0x8d, 0xd020); assembler.absolute(0x8d, 0xd021)
    assembler.absolute(0x20, "blank_lights"); assembler.byte(0x60)
    assembler.label("mode_lights")
    assembler.imm(0xa9, 0); assembler.absolute(0x8d, 0xd020); assembler.absolute(0x8d, 0xd021)
    assembler.absolute(0x4c, "draw_lights")
    assembler.label("mode_border")
    assembler.absolute(0x20, "blank_lights"); assembler.absolute(0x4c, "slow_colour")
    assembler.label("mode_both")
    assembler.absolute(0x20, "draw_lights"); assembler.absolute(0x4c, "slow_colour")

    def add_text(label: str, value: str) -> None:
        assembler.label(label)
        assembler.byte(*(screen_code(ch) for ch in value), 0)

    add_text("logo", "MIDI2SID")
    clean_title = " ".join(title.upper().replace("_", " ").split())[:38] or "UNTITLED"
    add_text("title", clean_title)
    add_text("voice1", "VOICE 1")
    add_text("voice2", "VOICE 2")
    add_text("voice3", "VOICE 3")
    add_text("keys1", "1 SAFE  2 LIGHTS  3 BORDER")
    add_text("keys2", "4 BOTH  5 DEMO")
    assembler.label("events")
    event_address = assembler.pc
    assembler.byte(*packed_events)

    code = bytearray(assembler.resolve())
    for offset, value in (
        (data_lo_at, event_address & 0xff), (data_hi_at, event_address >> 8),
        (restart_lo_at, event_address & 0xff), (restart_hi_at, event_address >> 8),
    ):
        code[offset] = value
    return bytes(code)


def build_prg(frames: list[bytes], output: str | Path, title: str = "MIDI2SID", video: str = "pal") -> Path:
    if video not in ("pal", "ntsc"):
        raise ValueError("video must be pal or ntsc")
    events = encode_events(frames)
    packed_events = pack_lzss(events)
    payload = _basic_stub() + _player(title, packed_events, video)
    end_address = LOAD_ADDRESS + len(payload) - 1
    if end_address > MAX_END_ADDRESS:
        raise ValueError(
            f"song is too large for PRG v1 ({len(events)} raw / {len(packed_events)} packed event bytes; "
            f"ends at ${end_address:04x}). A pattern-bank backend is needed for this file."
        )
    result = Path(output)
    result.write_bytes(LOAD_ADDRESS.to_bytes(2, "little") + payload)
    return result
