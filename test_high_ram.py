"""Tests for the $E000-$FFF8 high-RAM extension for oversized songs.

$D000-$DFFF is VIC/SID/CIA I/O and can never hold data, but with the CPU
port value the player already sets, $E000-$FFFF is plain RAM. Packed song
events that overflow the $CFFF ceiling spill into it instead of the build
simply failing. These tests run the actual generated 6502 code through the
sim (no VICE available in this sandbox), not just the Python reference
encoder/decoder, since the bug this guards against - getpacked() clobbering
its return byte on a page-boundary crossing - only showed up there: it was
invisible to decode_events()/unpack_lzss() (which only exercise the Python
reference implementation) and to frame-level spot checks (it self-heals
within a couple of frames, since only the single byte read at the exact
crossing instant is wrong).
"""
import unittest

import build_prg
import sid_midi
from cpu6502_sim import CPU


def _melodic_notes(count):
    notes = []
    t = 0
    for i in range(count):
        pitch = 48 + (i * 3) % 30
        notes.append(sid_midi.Note(t, t + 90, pitch, 100, 0, 80))
        notes.append(sid_midi.Note(t, t + 90, pitch - 24, 90, 6, 34))
        t += 100
    return notes


def _run_boot(mem):
    cpu = CPU(mem, char_rom=bytes(4096))
    try:
        cpu.run(build_prg.ENTRY_ADDRESS, max_steps=2_000_000)
    except RuntimeError:
        pass  # expected: the raster-wait loop spins forever with no real VIC
    return cpu


def _assemble_with_labels(packed_events, title="T"):
    captured = {}
    original_resolve = build_prg.Assembler.resolve

    def resolve(self):
        captured["labels"] = dict(self.labels)
        return original_resolve(self)

    build_prg.Assembler.resolve = resolve
    try:
        code = build_prg._player(title, packed_events, "pal")
    finally:
        build_prg.Assembler.resolve = original_resolve
    return code, captured["labels"]


class HighRamExtensionTests(unittest.TestCase):
    def test_getbyte_reproduces_every_decompressed_byte_across_page_boundaries(self):
        # A modest song already crosses several 256-byte page boundaries in
        # its packed stream (no high-RAM spillover needed to hit this), so
        # this alone would have caught the getpacked() regression.
        frames = sid_midi.frames_for(_melodic_notes(50), 96, drums="off")
        events = build_prg.encode_events(frames)
        packed = build_prg.pack_lzss(events)
        self.assertGreater(len(packed), 256 * 2, "fixture too small to cross a page boundary")

        code, labels = _assemble_with_labels(packed)
        mem = bytearray(65536)
        mem[build_prg.ENTRY_ADDRESS:build_prg.ENTRY_ADDRESS + len(code)] = code
        _run_boot(mem)

        cpu = CPU(mem, char_rom=bytes(4096))
        return_to = 0x9600
        for i, expected in enumerate(events):
            cpu.push((return_to - 1) >> 8); cpu.push((return_to - 1) & 0xff)
            cpu.run(labels["getbyte"], max_steps=5000, stop_at={return_to})
            self.assertEqual(cpu.a, expected, f"decompressed byte {i} wrong")

    def test_large_song_spills_into_high_ram_and_matches_reference_decoder(self):
        frames = sid_midi.frames_for(_melodic_notes(1400), 96, drums="off")
        events = build_prg.encode_events(frames)
        packed = build_prg.pack_lzss(events)
        low_capacity = build_prg.MAX_END_ADDRESS - 0x57c0 + 1  # rough, just needs to overflow
        self.assertGreater(len(packed), low_capacity, "fixture too small to need high RAM")

        code, labels = _assemble_with_labels(packed, title="SPILL")
        mem = bytearray(65536)
        mem[build_prg.ENTRY_ADDRESS:build_prg.ENTRY_ADDRESS + len(code)] = code
        cpu = _run_boot(mem)

        update_addr = labels["update"]
        return_to = 0x9500
        reached_high_ram = False
        for i, expected in enumerate(frames):
            cpu.push((return_to - 1) >> 8); cpu.push((return_to - 1) & 0xff)
            cpu.run(update_addr, max_steps=5000, stop_at={return_to})
            if mem[0xfc] >= 0xe0:
                reached_high_ram = True
            actual = bytes(mem[0xd400:0xd400 + 25])
            self.assertEqual(actual, bytes(expected), f"frame {i} wrong")
        # The event pointer must actually have walked into $E000+ at some
        # point for the rest of this test to mean anything.
        self.assertTrue(reached_high_ram, "event pointer never reached the high-RAM region")

    def test_reset_and_nmi_vectors_point_somewhere_safe(self):
        # A real RESTORE keypress fires NMI no matter the I flag. If the
        # vector pointed into arbitrary song bytes instead of a safe RTI
        # stub, the machine would jam.
        frames = sid_midi.frames_for(_melodic_notes(1400), 96, drums="off")
        packed = build_prg.pack_lzss(build_prg.encode_events(frames))
        code, labels = _assemble_with_labels(packed, title="SPILL")
        mem = bytearray(65536)
        mem[build_prg.ENTRY_ADDRESS:build_prg.ENTRY_ADDRESS + len(code)] = code

        stub = labels["vector_stub"]
        self.assertEqual(mem[stub], 0x40)  # RTI
        self.assertEqual(mem[0xfffa] | (mem[0xfffb] << 8), stub)          # NMI
        self.assertEqual(mem[0xfffc] | (mem[0xfffd] << 8), build_prg.ENTRY_ADDRESS)  # RESET
        self.assertEqual(mem[0xfffe] | (mem[0xffff] << 8), stub)          # IRQ

    def test_song_too_large_even_for_high_ram_raises_clear_error(self):
        frames = sid_midi.frames_for(_melodic_notes(1600), 96, drums="off")
        packed = build_prg.pack_lzss(build_prg.encode_events(frames))
        with self.assertRaises(ValueError) as ctx:
            build_prg._player("TOO BIG", packed, "pal")
        self.assertIn("too large", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
