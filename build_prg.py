#!/usr/bin/env python3
"""Build a self-running C64 PRG containing a 50/60 Hz SID event player."""
from __future__ import annotations

from pathlib import Path

LOAD_ADDRESS = 0x0801
ENTRY_ADDRESS = 0x080d
EVENT_DELAY = 0xfd
EVENT_FRAME = 0xfe
EVENT_END = 0xff
# $f1-$f7 encode a gate-off mask for SID voices 1-3. The player keeps parsing
# the same video frame, so updated frequency/ADSR registers and gate-on writes
# follow without adding a 20 ms silent frame.
EVENT_RETRIGGER_BASE = 0xf0
MAX_END_ADDRESS = 0xcfff
SCOPE_WIDTH = 12

# VIC-II hi-res bitmap layout. $2000 is forced: it is the only 8K-aligned
# bitmap base inside bank 0 that doesn't collide with zero page/stack/this
# program (the other choice, $0000, is where the program itself lives).
BITMAP_BASE = 0x2000
BITMAP_SIZE = 8000
INK_PAPER = 0xd0  # screen RAM nibble: light green (13) ink, black (0) paper
SCOPE_TRACE_BYTES = SCOPE_WIDTH * 8  # one contiguous bitmap block per trace
WAVE_PHASES = 16


def cell_addr(row: int, col: int) -> int:
    """C64 hi-res bitmap address of an 8x8 character cell's first byte.

    Unlike a linear framebuffer, VIC-II bitmap memory is laid out cell by
    cell: all 8 scanline bytes of one column live in one contiguous block,
    so drawing/blitting a whole cell (or a run of adjacent cells) is a
    single sequential copy with no per-scanline addressing needed.
    """
    return BITMAP_BASE + row * 320 + col * 8


def _wave_row(x: int, phase: int, period: int = 48, phases: int = WAVE_PHASES) -> int:
    offset = (x + phase * (period // phases)) % period
    half = period // 2
    return round(offset / half * 7) if offset < half else round((period - offset) / half * 7)


def _wave_phase_bytes(phase: int) -> bytes:
    """One SCOPE_TRACE_BYTES picture: a travelling triangle-wave line.

    Same shared-waveform idea MIDI2AY uses for its Spectrum oscilloscope
    (one phase-cycling picture reused by every channel), reimplemented for
    the C64's per-cell bitmap addressing: each column's 8 bytes are stored
    together so the whole trace is one flat block copy at render time.
    """
    out = bytearray(SCOPE_TRACE_BYTES)
    for col in range(SCOPE_WIDTH):
        column = bytearray(8)
        for x_in_col in range(8):
            y = _wave_row(col * 8 + x_in_col, phase)
            column[y] |= 0x80 >> x_in_col
        out[col * 8:col * 8 + 8] = bytes(column)
    return bytes(out)


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

    def word_label(self, label: str) -> None:
        """Emit a label's resolved address as two raw data bytes (no opcode)."""
        self.fixups.append((len(self.code), label, "absolute"))
        self.byte(0, 0)

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
        retrigger_mask = getattr(frame, "retrigger_mask", 0)
        changes = [(register, value) for register, value in enumerate(frame) if value != previous[register]]
        if not changes and not retrigger_mask:
            delay += 1
            continue
        flush_delay()
        controls = (4, 11, 18)
        # Release GATE before changing waveform or retriggering a new note. The
        # SID starts an ADSR envelope only on a 0->1 gate transition.
        forced_controls: set[int] = set()
        gate_off_mask = 0
        for voice, control in enumerate(controls):
            old, new = previous[control], frame[control]
            retrigger = bool(retrigger_mask & (1 << voice))
            if old & 1 and new & 1 and (retrigger or (old & 0xf0) != (new & 0xf0)):
                gate_off_mask |= 1 << voice
                forced_controls.add(control)
        if gate_off_mask:
            output.append(EVENT_RETRIGGER_BASE | gate_off_mask)
        # Frequency, pulse width and ADSR must be ready before GATE rises. The
        # old encoder wrote registers numerically, which put control/GATE ahead
        # of ADSR and let a new note begin with the previous patch's envelope.
        for register, value in changes:
            if register not in controls:
                output.extend((register, value))
        for register, value in changes:
            if register in controls:
                output.extend((register, value))
        for control in sorted(forced_controls):
            if all(register != control for register, _ in changes):
                output.extend((control, frame[control]))
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
        elif EVENT_RETRIGGER_BASE < command < EVENT_DELAY:
            # Retriggers are transient writes within a video frame; the final
            # decoded snapshot still has the gate raised by the later control
            # register write.
            continue
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
    return {
        " ": 32, "-": 45, ".": 46, "/": 47, ":": 58,
        "<": 60, ">": 62,
    }.get(character, 32)


def _basic_stub() -> bytes:
    # 10 SYS2061, followed by the zero next-line pointer. Machine code starts
    # at $080d (2061), immediately after this standard BASIC v2 launcher.
    return bytes((0x0b, 0x08, 0x0a, 0x00, 0x9e, 0x32, 0x30, 0x36, 0x31, 0x00, 0x00, 0x00))


def _player(title: str, packed_events: bytes, video: str) -> bytes:
    assembler = Assembler(ENTRY_ADDRESS)
    zp_phase1, zp_phase2, zp_phase3 = 0xee, 0xef, 0xf0
    zp_colour, zp_frame, zp_mode = 0xf2, 0xf3, 0xf4
    zp_register, zp_ring = 0xf5, 0xf6
    zp_flags, zp_bits = 0xf7, 0xf8
    zp_copy_len, zp_copy_pos = 0xf9, 0xfa
    zp_lo, zp_hi, zp_wait = 0xfb, 0xfc, 0xfd
    # Font-blit machinery: a source (ROM glyph) pointer, a destination
    # (bitmap cell) pointer, and a scratch byte, all reused later by the
    # per-frame scope render once the one-off font copy at boot is done.
    zp_str_lo, zp_str_hi = 0xe0, 0xe1
    zp_dst_lo, zp_dst_hi = 0xe2, 0xe3
    zp_src_lo, zp_src_hi = 0xe4, 0xe5
    zp_tmp = 0xe6

    # The whole display is one permanent VIC-II hi-res bitmap (no raster
    # split): title, labels and the oscilloscope traces are all pixels in
    # one buffer, the same way the Spectrum player's screen always is.
    # Static label text is rendered at boot by copying real glyph shapes
    # out of the character ROM (bank-switched in briefly via the $01 CPU
    # port) rather than a hand-authored font, since there is no way to
    # visually proof-read a hand-drawn font in this build environment but
    # the ROM's shapes are correct by construction.
    #
    # Label/data tables are emitted first so their addresses are already
    # known (as plain Python ints) by the time the init code below needs
    # to reference them; a leading JMP skips over this data block so
    # execution still starts at the real entry point.
    assembler.absolute(0x4c, "real_start")

    def add_text(label: str, value: str) -> None:
        assembler.label(label)
        assembler.byte(*(screen_code(ch) for ch in value), 0)

    add_text("logo", "MIDI2SID")
    clean_title = " ".join(title.upper().replace("_", " ").split())[:38] or "UNTITLED"
    add_text("title", clean_title)
    add_text("voice1", "LEAD")
    add_text("voice2", "BACKING")
    add_text("voice3", "BASS/DRUM")
    add_text("keys1", "1 SAFE  2 SCOPE  3 BORDER")
    add_text("keys2", "4 BOTH  5 DEMO")

    for phase in range(WAVE_PHASES):
        assembler.label(f"wave_phase_{phase}")
        assembler.byte(*_wave_phase_bytes(phase))
    wave_ptr_lo = assembler.pc
    assembler.label("wave_ptr_table")
    for phase in range(WAVE_PHASES):
        assembler.word_label(f"wave_phase_{phase}")
    wave_ptr_hi = wave_ptr_lo + 1

    label_rows = {
        "logo": (0, 16), "title": (2, 1),
        "voice1": (5, 2), "voice2": (9, 2), "voice3": (13, 2),
        "keys1": (22, 1), "keys2": (23, 1),
    }
    scope_rows = {0: 6, 1: 10, 2: 14}
    scope_voices = [
        (0xd404, 0xd401, zp_phase1, cell_addr(scope_rows[0], 2)),
        (0xd40b, 0xd408, zp_phase2, cell_addr(scope_rows[1], 2)),
        (0xd412, 0xd40f, zp_phase3, cell_addr(scope_rows[2], 2)),
    ]

    assembler.label("real_start")
    assembler.byte(0x78)                         # SEI
    # Bank switch: CHAREN=0 makes the character ROM visible at $D000-$DFFF
    # for the CPU (this also hides VIC/SID/CIA I/O for as long as it's set,
    # so nothing below may touch $D000-$DFFF until it's restored).
    assembler.imm(0xa9, 0x31); assembler.zp(0x85, 0x01)
    for label_name, (row, col) in label_rows.items():
        label_addr = assembler.labels[label_name]
        dest = cell_addr(row, col)
        assembler.imm(0xa9, label_addr & 0xff); assembler.zp(0x85, zp_str_lo)
        assembler.imm(0xa9, label_addr >> 8); assembler.zp(0x85, zp_str_hi)
        assembler.imm(0xa9, dest & 0xff); assembler.zp(0x85, zp_dst_lo)
        assembler.imm(0xa9, dest >> 8); assembler.zp(0x85, zp_dst_hi)
        assembler.absolute(0x20, "blit_label")
    # Restore normal I/O visibility (SID/VIC/CIA) for the rest of the program.
    assembler.imm(0xa9, 0x35); assembler.zp(0x85, 0x01)

    assembler.imm(0xa9, 0); assembler.absolute(0x8d, 0xd020); assembler.absolute(0x8d, 0xd021)
    # Every 8x8 cell's ink/paper nibble, uniform for now (per-cell colour
    # only matters once artwork gives cells different colours).
    assembler.imm(0xa2, 0)
    assembler.imm(0xa9, INK_PAPER)
    assembler.label("fill_screen_colours")
    for address in (0x0400, 0x0500, 0x0600, 0x0700):
        assembler.absolute(0x9d, address)
    assembler.byte(0xe8); assembler.branch(0xd0, "fill_screen_colours")
    # Hi-res bitmap mode: bitmap at $2000, screen (ink/paper) at $0400.
    assembler.imm(0xa9, 0x3b); assembler.absolute(0x8d, 0xd011)
    assembler.imm(0xa9, 0x18); assembler.absolute(0x8d, 0xd018)

    assembler.imm(0xa2, 24); assembler.imm(0xa9, 0)
    assembler.label("clear_sid")
    assembler.absolute(0x9d, 0xd400)
    assembler.byte(0xca); assembler.branch(0x10, "clear_sid")
    assembler.imm(0xa9, 15); assembler.absolute(0x8d, 0xd418)
    assembler.imm(0xa9, 0)
    for address in (
        zp_phase1, zp_phase2, zp_phase3, zp_colour, zp_frame, zp_ring,
        zp_flags, zp_bits, zp_copy_len, zp_copy_pos, zp_wait,
    ):
        assembler.zp(0x85, address)
    # The pitch-reactive scopes are safe and useful, so start in mode 2.
    assembler.imm(0xa9, 2); assembler.zp(0x85, zp_mode)
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
    assembler.imm(0xc9, EVENT_RETRIGGER_BASE + 1); assembler.branch(0x90, "register_write")
    assembler.imm(0xc9, EVENT_DELAY); assembler.branch(0x90, "retrigger")
    assembler.label("register_write")
    # The streaming decompressor uses X for its ring buffer, so retain the
    # target SID register in zero page while fetching the value byte.
    assembler.zp(0x85, zp_register)
    assembler.absolute(0x20, "getbyte")
    assembler.zp(0xa6, zp_register)
    assembler.absolute(0x9d, 0xd400)
    assembler.absolute(0x4c, "parse")
    assembler.label("retrigger")
    assembler.zp(0x85, zp_register)
    for bit, control, done in (
        (1, 0xd404, "retrigger2"),
        (2, 0xd40b, "retrigger3"),
        (4, 0xd412, "retrigger_done"),
    ):
        assembler.zp(0xa5, zp_register); assembler.imm(0x29, bit); assembler.branch(0xf0, done)
        assembler.absolute(0xad, control); assembler.imm(0x29, 0xfe); assembler.absolute(0x8d, control)
        assembler.label(done)
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
    for address in (
        zp_phase1, zp_phase2, zp_phase3, zp_ring, zp_flags, zp_bits,
        zp_copy_len, zp_copy_pos, zp_wait,
    ):
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

    # Each voice's trace is a shared travelling-triangle picture (like
    # MIDI2AY's WAVE_TABLE) copied wholesale into its bitmap cell row: since
    # a run of adjacent bitmap columns is one contiguous block (see
    # cell_addr), the whole SCOPE_TRACE_BYTES trace is a single flat copy,
    # no per-character addressing needed. Phase still advances at ~12.5 Hz
    # with the SID frequency high byte nudging its step, exactly as before.
    assembler.label("draw_scopes")
    for index, (control, frequency_hi, phase, dest) in enumerate(scope_voices):
        assembler.absolute(0xad, control); assembler.imm(0x29, 1)
        assembler.branch(0xf0, f"scope_clear_{index}")
        assembler.zp(0xa5, zp_frame); assembler.imm(0x29, 3)
        assembler.branch(0xd0, f"scope_render_{index}")
        assembler.absolute(0xad, frequency_hi)
        for _ in range(6):
            assembler.byte(0x4a)                  # LSR A
        assembler.byte(0x18); assembler.imm(0x69, 1)
        assembler.zp(0x65, phase); assembler.imm(0x29, 0x0f)
        assembler.zp(0x85, phase)
        assembler.label(f"scope_render_{index}")
        assembler.zp(0xa5, phase)
        assembler.byte(0x0a)                      # ASL A (word index into ptr table)
        assembler.byte(0xa8)                      # TAY
        assembler.absolute(0xb9, wave_ptr_lo)     # LDA wave_ptr_table,Y
        assembler.zp(0x85, zp_src_lo)
        assembler.absolute(0xb9, wave_ptr_hi)     # LDA wave_ptr_table+1,Y
        assembler.zp(0x85, zp_src_hi)
        assembler.imm(0xa0, 0)
        assembler.label(f"scope_copy_{index}")
        assembler.byte(0xb1, zp_src_lo)           # LDA (zp_src),Y
        assembler.absolute(0x99, dest)            # STA dest,Y
        assembler.byte(0xc8)                      # INY
        assembler.imm(0xc0, SCOPE_TRACE_BYTES)
        assembler.branch(0xd0, f"scope_copy_{index}")
        assembler.absolute(0x4c, f"scope_done_{index}")
        assembler.label(f"scope_clear_{index}")
        assembler.imm(0xa2, 0); assembler.imm(0xa9, 0)
        assembler.label(f"scope_blank_{index}")
        assembler.absolute(0x9d, dest)            # STA dest,X
        assembler.byte(0xe8)
        assembler.imm(0xe0, SCOPE_TRACE_BYTES)
        assembler.branch(0xd0, f"scope_blank_{index}")
        assembler.label(f"scope_done_{index}")
    assembler.byte(0x60)

    assembler.label("blank_scopes")
    assembler.imm(0xa9, 0)
    for index, (_, _, _, dest) in enumerate(scope_voices):
        assembler.imm(0xa2, 0)
        assembler.label(f"blank_all_{index}")
        assembler.absolute(0x9d, dest)
        assembler.byte(0xe8)
        assembler.imm(0xe0, SCOPE_TRACE_BYTES)
        assembler.branch(0xd0, f"blank_all_{index}")
    assembler.byte(0x60)

    assembler.label("slow_colour")
    assembler.zp(0xa5, zp_frame); assembler.imm(0x29, 0x0f); assembler.branch(0xd0, "slow_done")
    assembler.zp(0xe6, zp_colour); assembler.zp(0xa5, zp_colour); assembler.imm(0x29, 0x0f)
    assembler.absolute(0x8d, 0xd020)
    assembler.label("slow_done"); assembler.byte(0x60)

    assembler.label("visual")
    assembler.absolute(0x20, "scan_keys"); assembler.zp(0xe6, zp_frame)
    assembler.zp(0xa5, zp_mode); assembler.imm(0xc9, 1); assembler.branch(0xf0, "mode_safe")
    assembler.imm(0xc9, 2); assembler.branch(0xf0, "mode_scope")
    assembler.imm(0xc9, 3); assembler.branch(0xf0, "mode_border")
    assembler.imm(0xc9, 4); assembler.branch(0xf0, "mode_both")
    # Mode 5's old background-colour pulse wrote $D021, which has no visible
    # effect once the whole screen is bitmap (ink/paper comes from the per-
    # cell screen RAM nibble instead); it now matches mode 4.
    assembler.absolute(0x20, "draw_scopes"); assembler.absolute(0x4c, "slow_colour")
    assembler.label("mode_safe")
    assembler.imm(0xa9, 0); assembler.absolute(0x8d, 0xd020)
    assembler.absolute(0x20, "blank_scopes"); assembler.byte(0x60)
    assembler.label("mode_scope")
    assembler.imm(0xa9, 0); assembler.absolute(0x8d, 0xd020)
    assembler.absolute(0x4c, "draw_scopes")
    assembler.label("mode_border")
    assembler.absolute(0x20, "blank_scopes"); assembler.absolute(0x4c, "slow_colour")
    assembler.label("mode_both")
    assembler.absolute(0x20, "draw_scopes"); assembler.absolute(0x4c, "slow_colour")

    # Copies one glyph's 8 ROM scanline bytes into each bitmap cell of a
    # 0-terminated screencode string; zp_str/zp_dst are set by the caller.
    assembler.label("blit_label")
    assembler.label("blit_next_char")
    assembler.imm(0xa0, 0)
    assembler.byte(0xb1, zp_str_lo)               # LDA (zp_str),Y
    assembler.branch(0xf0, "blit_done")
    assembler.zp(0x85, zp_tmp)
    assembler.imm(0xa9, 0); assembler.zp(0x85, zp_src_hi)
    assembler.zp(0xa5, zp_tmp); assembler.zp(0x85, zp_src_lo)
    for _ in range(3):                            # zp_src = code * 8
        assembler.zp(0x06, zp_src_lo)              # ASL zp_src_lo
        assembler.zp(0x26, zp_src_hi)              # ROL zp_src_hi
    assembler.zp(0xa5, zp_src_hi)
    assembler.byte(0x18); assembler.imm(0x69, 0xd0)  # + $D000 (char ROM base)
    assembler.zp(0x85, zp_src_hi)
    assembler.imm(0xa0, 0)
    assembler.label("blit_copy_row")
    assembler.byte(0xb1, zp_src_lo)               # LDA (zp_src),Y
    assembler.byte(0x91, zp_dst_lo)                # STA (zp_dst),Y
    assembler.byte(0xc8)
    assembler.imm(0xc0, 8)
    assembler.branch(0xd0, "blit_copy_row")
    assembler.zp(0xa5, zp_dst_lo)
    assembler.byte(0x18); assembler.imm(0x69, 8)
    assembler.zp(0x85, zp_dst_lo)
    assembler.branch(0x90, "blit_dst_no_carry")
    assembler.zp(0xe6, zp_dst_hi)
    assembler.label("blit_dst_no_carry")
    assembler.zp(0xe6, zp_str_lo)
    assembler.branch(0xd0, "blit_next_char")
    assembler.zp(0xe6, zp_str_hi)
    assembler.absolute(0x4c, "blit_next_char")
    assembler.label("blit_done")
    assembler.byte(0x60)

    # The 8000-byte bitmap must start at $2000 (VIC-II 8K bitmap alignment);
    # pad the fixed code up to that boundary, then reserve the bitmap as a
    # blank canvas (labels/scope are drawn into it above, at runtime). The
    # packed song event stream continues right after it.
    if assembler.pc > BITMAP_BASE:
        raise ValueError(
            f"player code grew past ${BITMAP_BASE:04x} (now ${assembler.pc:04x}); "
            "the bitmap must start exactly there"
        )
    assembler.byte(*([0] * (BITMAP_BASE - assembler.pc)))
    assembler.label("bitmap")
    assembler.byte(*([0] * BITMAP_SIZE))

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
