import tempfile
import unittest
from pathlib import Path

import build_prg
import sid_midi


def midi_track(events):
    payload = b"".join(events) + b"\x00\xff\x2f\x00"
    return b"MTrk" + len(payload).to_bytes(4, "big") + payload


class SidSynthesisTests(unittest.TestCase):
    def test_all_general_midi_programs_are_mapped(self):
        self.assertEqual(len({sid_midi.family(n) for n in range(128)}), 16)
        self.assertEqual(sid_midi.family(0), "piano")
        self.assertEqual(sid_midi.family(80), "lead")
        self.assertEqual(sid_midi.family(127), "sfx")

    def test_pal_a4_frequency(self):
        self.assertEqual(sid_midi.sid_frequency(69), 7493)

    def test_smart_allocator_keeps_lead_and_bass(self):
        notes = [
            sid_midi.Note(0, 100, 36, 90, 0, 32),
            sid_midi.Note(0, 100, 60, 80, 0, 48),
            sid_midi.Note(0, 30, 72, 120, 0, 80),
            sid_midi.Note(0, 100, 76, 40, 0, 88),
        ]
        voices, _, _ = sid_midi._choose_voices(notes)
        self.assertEqual(voices[0].pitch, 72)
        self.assertEqual(voices[2].pitch, 36)

    def test_drum_steals_voice_three_and_uses_noise(self):
        notes = [
            sid_midi.Note(0, 96, 60, 100, 0, 0),
            sid_midi.Note(0, 2, 38, 127, 9, 0),
        ]
        frame = sid_midi.frames_for(notes, 96, drums="smart")[0]
        self.assertTrue(frame[18] & sid_midi.NOISE)
        self.assertTrue(frame[18] & sid_midi.GATE)
        self.assertEqual(frame[20] >> 4, 0)

    def test_filter_is_off_by_default(self):
        note = sid_midi.Note(0, 96, 60, 100, 0, 32)
        frames = sid_midi.frames_for([note], 96)
        self.assertTrue(all(frame[21:24] == bytes(3) and frame[24] == 15 for frame in frames))

    def test_format_one_programs_do_not_leak_between_tracks(self):
        header = b"MThd" + (6).to_bytes(4, "big") + b"\x00\x01\x00\x02\x00\x60"
        lead = midi_track((b"\x00\xc0\x50", b"\x00\x90\x3c\x64", b"\x60\x80\x3c\x00"))
        piano = midi_track((b"\x00\x90\x40\x64", b"\x60\x80\x40\x00"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "programs.mid"
            path.write_bytes(header + lead + piano)
            _, notes, _ = sid_midi.read_midi(path)
        self.assertEqual({note.program for note in notes}, {0, 80})


class PrgTests(unittest.TestCase):
    def test_event_round_trip_preserves_frames(self):
        first = bytes([1] + [0] * 23 + [15])
        second = bytes([2] + [0] * 23 + [15])
        source = [first, first, first, second]
        decoded = build_prg.decode_events(build_prg.encode_events(source))
        self.assertEqual(decoded[:4], source)
        self.assertEqual(decoded[-1][24], 0)

    def test_lzss_round_trip(self):
        source = (b"SID-SID-SID-" * 80) + bytes(range(255)) + b"\xff"
        packed = build_prg.pack_lzss(source)
        self.assertEqual(build_prg.unpack_lzss(packed), source)
        self.assertLess(len(packed), len(source))

    def test_waveform_change_is_preceded_by_gate_off(self):
        old = bytearray(25); old[4] = sid_midi.PULSE | sid_midi.GATE; old[24] = 15
        new = bytearray(old); new[4] = sid_midi.NOISE | sid_midi.GATE
        events = build_prg.encode_events([bytes(old), bytes(new)])
        marker = bytes((4, sid_midi.PULSE, 4, sid_midi.NOISE | sid_midi.GATE))
        self.assertIn(marker, events)

    def test_prg_has_load_address_and_basic_sys_stub(self):
        frame = bytes([0] * 24 + [15])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "tiny.prg"
            build_prg.build_prg([frame], output, "TINY")
            data = output.read_bytes()
        self.assertEqual(data[:2], b"\x01\x08")
        self.assertIn(b"2061", data[:16])
        self.assertIn(bytes([build_prg.screen_code(ch) for ch in "TINY"]), data)

    def test_bad_frame_size_is_rejected(self):
        with self.assertRaises(ValueError):
            build_prg.encode_events([bytes(24)])


if __name__ == "__main__":
    unittest.main()
