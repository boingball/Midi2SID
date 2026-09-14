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

    def test_sparse_chords_do_not_duplicate_sid_voices(self):
        one = [sid_midi.Note(0, 100, 60, 100, 0, 0)]
        two = one + [sid_midi.Note(0, 100, 72, 100, 0, 80)]
        one_voices, _, _ = sid_midi._choose_voices(one)
        two_voices, _, _ = sid_midi._choose_voices(two)
        self.assertEqual([note.pitch if note else None for note in one_voices], [60, None, None])
        self.assertEqual([note.pitch if note else None for note in two_voices], [72, None, 60])

    def test_whole_song_score_prefers_recurring_pipe_melody(self):
        notes = [sid_midi.Note(0, 2400, 83, 84, 7, 84)]
        for index in range(24):
            start = 3000 + index * 60
            notes.append(sid_midi.Note(start, start + 30, 79 + index % 5, 112, 5, 79))
            notes.append(sid_midi.Note(start, start + 55, 40 + index % 3, 104, 0, 35))
        self.assertEqual(sid_midi._select_lead_channel(notes), 5)

    def test_selected_melody_channel_beats_short_arpeggio(self):
        notes = [
            sid_midi.Note(0, 120, 84, 112, 5, 79),
            sid_midi.Note(0, 12, 96, 127, 2, 102),
            sid_midi.Note(0, 120, 40, 104, 0, 35),
        ]
        voices, _, _ = sid_midi._choose_voices(notes, lead_channel=5)
        self.assertEqual(voices[0].channel, 5)
        self.assertEqual(voices[2].channel, 0)
        self.assertIsNone(voices[1])

    def test_pipe_patch_is_bright_pulse_lead(self):
        self.assertEqual(sid_midi.PATCHES["pipe"].waveform, sid_midi.PULSE)
        self.assertGreaterEqual(sid_midi.PATCHES["pipe"].sustain, 12)

    def test_lead_sustain_is_mixed_above_backing_and_bass(self):
        note = sid_midi.Note(0, 120, 72, 112, 5, 79)
        levels = [
            sid_midi._patch_registers(note, 0, voice, sid_midi.PAL_SID_CLOCK, "tight")[6] >> 4
            for voice in range(3)
        ]
        self.assertGreater(levels[0], levels[1])
        self.assertGreater(levels[0], levels[2])

    def test_tight_feel_uses_fast_attack_and_velocity_floor(self):
        note = sid_midi.Note(0, 96, 60, 1, 0, 48)
        tight = sid_midi.frames_for([note], 96, feel="tight")[0]
        expressive = sid_midi.frames_for([note], 96, feel="expressive")[0]
        self.assertEqual(tight[5] >> 4, 0)
        self.assertEqual(expressive[5] >> 4, sid_midi.PATCHES["ensemble"].attack)
        self.assertGreater(tight[6] >> 4, expressive[6] >> 4)

    def test_tight_feel_keeps_pulse_width_stable(self):
        note = sid_midi.Note(0, 960, 60, 100, 0, 0)
        tight = sid_midi._patch_registers(note, 4, 0, sid_midi.PAL_SID_CLOCK, "tight")
        expressive = sid_midi._patch_registers(note, 4, 0, sid_midi.PAL_SID_CLOCK, "expressive")
        self.assertEqual(tight[2:4], [0x80, 0x06])
        self.assertNotEqual(tight[2:4], expressive[2:4])

    def test_new_same_waveform_note_sets_retrigger_mask(self):
        notes = [
            sid_midi.Note(0, 96, 60, 100, 0, 0),
            sid_midi.Note(96, 192, 62, 100, 0, 0),
        ]
        frames = sid_midi.frames_for(notes, 96)
        changed = [frame for frame in frames if frame.retrigger_mask]
        self.assertTrue(changed)
        self.assertTrue(changed[0].retrigger_mask & 1)

    def test_smart_drums_have_short_voice_stealing(self):
        self.assertEqual(sid_midi._drum_tail(42), 1)
        self.assertEqual(sid_midi._drum_tail(38), 2)
        self.assertEqual(sid_midi._drum_tail(51), 3)

    def test_fractional_midi_clock_does_not_accumulate_tempo_error(self):
        # 120 PPQN at 132 BPM is 5.28 ticks per PAL frame. A rounded five-tick
        # clock would produce 25 frames instead of the correct 24 here.
        note = sid_midi.Note(0, 120, 60, 100, 0, 0)
        frames = sid_midi.frames_for([note], 120, tempos=[(0, 454545)])
        self.assertEqual(len(frames), 24)

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
        marker = bytes((build_prg.EVENT_RETRIGGER_BASE | 1, 4, sid_midi.NOISE | sid_midi.GATE))
        self.assertIn(marker, events)

    def test_retrigger_writes_adsr_before_gate_on(self):
        first = bytearray(25); first[4] = sid_midi.PULSE | sid_midi.GATE; first[5] = 1; first[24] = 15
        second = bytearray(first); second[0] = 2; second[5] = 8
        frame = sid_midi.SidFrame(second, retrigger_mask=1)
        events = build_prg.encode_events([bytes(first), frame])
        marker = bytes((build_prg.EVENT_RETRIGGER_BASE | 1, 0, 2, 5, 8, 4, sid_midi.PULSE | sid_midi.GATE))
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

    def test_prg_contains_pitch_reactive_scope_ui(self):
        frame = bytes([0] * 24 + [15])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "scope.prg"
            build_prg.build_prg([frame], output, "SCOPE")
            data = output.read_bytes()
        self.assertIn(bytes(build_prg.screen_code(ch) for ch in "1 SAFE  2 SCOPE"), data)
        self.assertIn(bytes(build_prg.screen_code(ch) for ch in "..-->>>--..<<<--"), data)
        self.assertIn(bytes(build_prg.screen_code(ch) for ch in "BASS/DRUM"), data)

    def test_bad_frame_size_is_rejected(self):
        with self.assertRaises(ValueError):
            build_prg.encode_events([bytes(24)])


if __name__ == "__main__":
    unittest.main()
