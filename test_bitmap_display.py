"""Regression tests for the VIC-II hi-res bitmap display.

There is no C64 emulator (VICE or otherwise) available to check the
hand-assembled 6502/VIC-II code visually, so these tests run the generated
init code through a small pure-Python 6502 interpreter (cpu6502_sim.py) and
check the resulting memory image directly: that bitmap mode actually gets
enabled, that each label's glyphs land in the right bitmap cells, and that
the oscilloscope trace renders real (non-blank) pixels for a gated voice and
stays blank for a silent one.
"""
import unittest

import build_prg
import sid_midi
from cpu6502_sim import CPU


def _assemble_with_labels(title="Test Song", video="pal"):
    reg_a = bytearray(25)
    reg_a[0] = 0
    reg_a[1] = 0x10
    reg_a[4] = sid_midi.TRIANGLE | sid_midi.GATE
    frames = [sid_midi.SidFrame(reg_a, 0), sid_midi.SidFrame(bytearray(reg_a), 0)]
    packed = build_prg.pack_lzss(build_prg.encode_events(frames))

    captured = {}
    original_resolve = build_prg.Assembler.resolve

    def resolve(self):
        captured["labels"] = dict(self.labels)
        return original_resolve(self)

    build_prg.Assembler.resolve = resolve
    try:
        code = build_prg._player(title, packed, video)
    finally:
        build_prg.Assembler.resolve = original_resolve
    return code, captured["labels"]


def _fake_char_rom() -> bytes:
    # Each glyph's 8 bytes are its own screencode, repeated: lets tests
    # verify addressing (right glyph -> right cell) without needing a real
    # character ROM dump. Kept as a separate read-only buffer (not part of
    # `mem`) so a write to a $D000-$DFFF I/O register while CHAREN is high
    # can't appear to "corrupt" it, matching real hardware where character
    # ROM is physically read-only.
    rom = bytearray(4096)
    for screencode in range(256):
        base = screencode * 8
        for row in range(8):
            rom[base + row] = screencode
    return bytes(rom)


def _boot_memory(code, fake_rom=True):
    mem = bytearray(65536)
    mem[build_prg.ENTRY_ADDRESS:build_prg.ENTRY_ADDRESS + len(code)] = code
    writes = []
    cpu = CPU(mem, char_rom=_fake_char_rom() if fake_rom else None)
    cpu.io_hook = lambda addr, value: writes.append((addr, value))
    try:
        cpu.run(build_prg.ENTRY_ADDRESS, max_steps=500_000)
    except RuntimeError:
        pass  # expected: the raster-wait loop spins forever with no real VIC
    return mem, writes


class BitmapDisplayTests(unittest.TestCase):
    def test_boot_enables_hires_bitmap_mode_and_restores_io(self):
        code, _ = _assemble_with_labels()
        _, writes = _boot_memory(code)
        self.assertIn((0xd011, 0x3b), writes)   # BMM|DEN|RSEL, bitmap on
        self.assertIn((0xd018, 0x18), writes)   # bitmap $2000 / screen $0400
        self.assertEqual(writes[0], (0x01, 0x35))  # I/O fixed visible first
        char_rom_in = writes.index((0x01, 0x31))    # then char ROM banked in
        char_rom_out = writes.index((0x01, 0x35), char_rom_in)  # and restored after
        self.assertGreater(char_rom_out, char_rom_in)

    def test_labels_blit_to_the_right_bitmap_cells(self):
        code, _ = _assemble_with_labels(title="Test Song")
        mem, _ = _boot_memory(code)
        placements = {
            "MIDI2SID": (0, 16),
            "TEST SONG": (2, 1),
            "LEAD": (5, 2),
            "BACKING": (9, 2),
            "BASS/DRUM": (13, 2),
        }
        for text, (row, col) in placements.items():
            for i, ch in enumerate(text):
                dest = build_prg.cell_addr(row, col + i)
                expected = bytes([build_prg.screen_code(ch)] * 8)
                self.assertEqual(
                    mem[dest:dest + 8], expected,
                    f"{text!r} char {i} ({ch!r}) not at row {row} col {col + i}",
                )

    def test_scope_draws_pixels_for_a_gated_voice_and_blanks_a_silent_one(self):
        code, labels = _assemble_with_labels()
        mem, _ = _boot_memory(code)
        mem[0xd404] = sid_midi.TRIANGLE | 1  # voice 1 gated on
        mem[0xd401] = 0x20
        mem[0xd40b] = 0  # voice 2 gate off
        mem[0xd412] = 0  # voice 3 gate off
        mem[0xf3] = 0    # zp_frame: force a phase recompute this call

        cpu = CPU(mem)
        return_to = 0x9000
        cpu.push((return_to - 1) >> 8)
        cpu.push((return_to - 1) & 0xff)
        cpu.run(labels["draw_scopes"], max_steps=200_000, stop_at={return_to})

        dest1 = build_prg.cell_addr(6, 2)
        dest2 = build_prg.cell_addr(10, 2)
        dest3 = build_prg.cell_addr(14, 2)
        self.assertTrue(any(mem[dest1:dest1 + build_prg.SCOPE_TRACE_BYTES]))
        self.assertFalse(any(mem[dest2:dest2 + build_prg.SCOPE_TRACE_BYTES]))
        self.assertFalse(any(mem[dest3:dest3 + build_prg.SCOPE_TRACE_BYTES]))

    def test_code_fits_below_the_bitmap(self):
        code, labels = _assemble_with_labels()
        self.assertEqual(labels["bitmap"], build_prg.BITMAP_BASE)
        # The four waveform-shape picture sets + their pointer table live
        # right after the bitmap, ahead of the variable-length song events.
        wave_data_bytes = len(build_prg.WAVEFORMS) * build_prg.WAVE_PHASES * (
            build_prg.SCOPE_TRACE_BYTES + 2
        )
        self.assertEqual(
            labels["events"],
            build_prg.BITMAP_BASE + build_prg.BITMAP_SIZE + wave_data_bytes,
        )

    def test_scope_shape_matches_each_voices_own_waveform(self):
        code, labels = _assemble_with_labels()
        mem, _ = _boot_memory(code)

        def render(control_byte, freq_hi, phase_zp_addr, control_reg, freq_reg, dest):
            mem[control_reg] = control_byte
            mem[freq_reg] = freq_hi
            mem[phase_zp_addr] = 0
            mem[0xf3] = 0  # zp_frame: force phase recompute
            cpu = CPU(mem)
            return_to = 0x9000
            cpu.push((return_to - 1) >> 8)
            cpu.push((return_to - 1) & 0xff)
            cpu.run(labels["draw_scopes"], max_steps=200_000, stop_at={return_to})
            return bytes(mem[dest:dest + build_prg.SCOPE_TRACE_BYTES])

        dest1 = build_prg.cell_addr(6, 2)
        triangle_trace = render(sid_midi.TRIANGLE | 1, 0x20, 0xee, 0xd404, 0xd401, dest1)
        saw_trace = render(sid_midi.SAW | 1, 0x20, 0xee, 0xd404, 0xd401, dest1)
        pulse_trace = render(sid_midi.PULSE | 1, 0x20, 0xee, 0xd404, 0xd401, dest1)
        noise_trace = render(sid_midi.NOISE | 1, 0x20, 0xee, 0xd404, 0xd401, dest1)

        traces = [triangle_trace, saw_trace, pulse_trace, noise_trace]
        for a in range(len(traces)):
            for b in range(a + 1, len(traces)):
                self.assertNotEqual(traces[a], traces[b], "different waveforms drew identical traces")
        for trace in traces:
            self.assertTrue(any(trace))

    def test_pulse_voices_differ_by_their_own_live_pulse_width(self):
        # Over half the GM patch table maps to PULSE, so without reading
        # each voice's own pulse-width register they'd all draw the exact
        # same picture and only the phase nudge would tell them apart -
        # easy to miss on a real song. A narrow-duty bass patch and a
        # near-square lead patch must render visibly different traces even
        # with the same control byte, frequency and phase.
        code, labels = _assemble_with_labels()
        mem, _ = _boot_memory(code)

        def render(pulse_width_hi):
            mem[0xd404] = sid_midi.PULSE | 1  # voice 1 gated on, PULSE
            mem[0xd401] = 0x20                # frequency-hi
            mem[0xd403] = pulse_width_hi      # voice 1's own pulse-width-hi
            mem[0xee] = 0                     # zp_phase1
            mem[0xf3] = 0                     # zp_frame: force phase recompute
            cpu = CPU(mem)
            return_to = 0x9000
            cpu.push((return_to - 1) >> 8)
            cpu.push((return_to - 1) & 0xff)
            cpu.run(labels["draw_scopes"], max_steps=200_000, stop_at={return_to})
            dest = build_prg.cell_addr(6, 2)
            return bytes(mem[dest:dest + build_prg.SCOPE_TRACE_BYTES])

        narrow_trace = render(0x00)
        wide_trace = render(0x0f)
        self.assertNotEqual(narrow_trace, wide_trace)

    def test_higher_pitched_voice_advances_its_phase_faster(self):
        # A shift of >>6 on an 8-bit frequency-hi byte left almost every
        # musically useful note with the same nudge, so all three scope
        # traces advanced in lockstep regardless of pitch. Confirm a bass
        # note and a lead note now produce different phase steps.
        code, labels = _assemble_with_labels()
        mem, _ = _boot_memory(code)

        def phase_step(freq_hi):
            mem[0xd404] = sid_midi.TRIANGLE | 1  # voice 1 gated on
            mem[0xd401] = freq_hi
            mem[0xee] = 0  # zp_phase1 starts at 0
            mem[0xf3] = 0  # zp_frame: force a phase recompute this call
            cpu = CPU(mem)
            return_to = 0x9000
            cpu.push((return_to - 1) >> 8)
            cpu.push((return_to - 1) & 0xff)
            cpu.run(labels["draw_scopes"], max_steps=200_000, stop_at={return_to})
            return mem[0xee]

        bass_step = phase_step(4)    # ~MIDI 36
        lead_step = phase_step(69)   # ~MIDI 84
        self.assertNotEqual(bass_step, lead_step)
        self.assertGreater(lead_step, bass_step)


if __name__ == "__main__":
    unittest.main()
