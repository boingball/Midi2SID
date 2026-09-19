import tempfile
import unittest
from fractions import Fraction
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
        # A lone long high note must not beat a recurring monophonic melody.
        # Use enough melody events to model a complete song rather than a tiny
        # phrase; the supplied Popcorn MIDI has 449 notes on its lead channel.
        notes = [sid_midi.Note(0, 2400, 83, 84, 7, 84)]
        for index in range(300):
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

    def test_reserved_lead_voice_rejects_arpeggio_during_melody_gap(self):
        notes = [
            sid_midi.Note(120, 132, 96, 127, 2, 102),
            sid_midi.Note(100, 180, 40, 104, 0, 35),
        ]
        voices, _, _ = sid_midi._choose_voices(
            notes, lead_channel=5, reserve_lead=True
        )
        self.assertIsNone(voices[0])
        self.assertEqual(voices[2].channel, 0)

    def test_lead_voice_fallback_remains_available_outside_melody_window(self):
        notes = [
            sid_midi.Note(0, 120, 72, 100, 6, 17),
            sid_midi.Note(0, 120, 40, 100, 0, 35),
        ]
        voices, _, _ = sid_midi._choose_voices(
            notes, lead_channel=5, reserve_lead=False
        )
        self.assertIsNotNone(voices[0])

    def test_reserve_lead_allows_fallback_during_a_long_lead_absence(self):
        # A synth lead riff (channel 5) that only plays briefly at the start
        # and end of a song, with a guitar solo (channel 7) filling a long
        # middle stretch, reproduces a real full-band arrangement: the lead
        # channel is still correctly identified as the song's melody, but
        # voice one must not go silent for the whole gap where it isn't
        # playing while good fallback material is available. Reserving the
        # full first-note-to-last-note span (rather than just the lead's own
        # notes and short rests between them) silenced voice one for the
        # entire solo instead of falling back to it.
        lead_notes = [
            sid_midi.Note(0, 40, 76, 110, 5, 80),
            sid_midi.Note(40, 100, 79, 110, 5, 80),
            sid_midi.Note(3000, 3040, 76, 110, 5, 80),
            sid_midi.Note(3040, 3100, 79, 110, 5, 80),
        ]
        solo_notes = [
            sid_midi.Note(start, start + 90, 60 + (start // 100) % 5, 100, 7, 30)
            for start in range(1000, 2000, 100)
        ]
        notes = lead_notes + solo_notes
        self.assertEqual(sid_midi._select_lead_channel(notes), 5)
        frames = sid_midi.frames_for(notes, 120, tempos=[(0, 500_000)])
        mid_tick = 1500
        tempo = 500_000
        ticks_per_frame = Fraction(120 * 1_000_000, tempo * 50)
        mid_frame = mid_tick * ticks_per_frame.denominator // ticks_per_frame.numerator
        self.assertTrue(
            frames[mid_frame][4] & sid_midi.GATE,
            "voice one went silent during the guitar solo instead of falling back",
        )

    def test_secondary_lead_wins_intro_over_loud_backing_chords(self):
        # A song whose main lead (channel 5, a real GM lead patch) only plays
        # late, with a staccato brass riff (channel 2) carrying an intro that
        # loud, sustained ensemble backing chords (channel 3) overlap: this
        # reproduces "The Final Countdown"'s intro, where the generic
        # per-frame fallback let the backing chords outscore and replace the
        # actual intro melody on voice one every other bar.
        primary_notes = [
            sid_midi.Note(5000 + i * 100, 5000 + i * 100 + 90, 72, 110, 5, 80)
            for i in range(20)
        ]
        secondary_notes = [
            sid_midi.Note(i * 100, i * 100 + 40, 70, 110, 2, 61) for i in range(10)
        ]
        backing_notes = [
            sid_midi.Note(i * 100, i * 100 + 100, pitch, 100, 3, 50)
            for i in range(10) for pitch in (60, 64, 67)
        ]
        notes = primary_notes + secondary_notes + backing_notes
        primary = sid_midi._select_lead_channel(notes)
        self.assertEqual(primary, 5)
        self.assertEqual(sid_midi._select_secondary_lead_channel(notes, primary), 2)

        frames = sid_midi.frames_for(notes, 120, tempos=[(0, 500_000)])
        tempo = 500_000
        ticks_per_frame = Fraction(120 * 1_000_000, tempo * 50)
        num, den = ticks_per_frame.numerator, ticks_per_frame.denominator
        for tick in range(0, 40, 4):  # inside the riff's first note, 0-40
            frame = tick * den // num
            control = frames[frame][4]
            if control & sid_midi.GATE:
                self.assertEqual(control & 0xf0, sid_midi.PATCHES["brass"].waveform)

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

    def test_note_off_preserves_waveform_and_sid_release(self):
        note = sid_midi.Note(0, 12, 83, 112, 5, 79)
        frames = sid_midi.frames_for([note], 96, feel="tight")
        gated = [index for index, frame in enumerate(frames) if frame[4] & sid_midi.GATE]
        self.assertTrue(gated)
        release_frame = frames[gated[-1] + 1]
        self.assertEqual(release_frame[4] & sid_midi.GATE, 0)
        self.assertTrue(release_frame[4] & sid_midi.PULSE)
        self.assertEqual(release_frame[6] & 0x0f, 3)

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

    def test_congas_and_bongos_get_tuned_pulse_not_generic_noise(self):
        # Latin GM drum kits lean heavily on 60-66/78-79; previously anything
        # outside the tom/kick/snare/cymbal lists collapsed to flat noise.
        for pitch in (60, 62, 64, 65, 78):
            registers = sid_midi._drum_registers(sid_midi.Note(0, 2, pitch, 100, 9, 0), 0, sid_midi.PAL_SID_CLOCK)
            self.assertEqual(registers[4] & 0xf0, sid_midi.PULSE)

    def test_agogo_and_ride_bell_are_tuned_like_cowbell(self):
        for pitch in (53, 67, 68):
            registers = sid_midi._drum_registers(sid_midi.Note(0, 2, pitch, 100, 9, 0), 0, sid_midi.PAL_SID_CLOCK)
            self.assertEqual(registers[4] & 0xf0, sid_midi.PULSE)
            self.assertEqual(sid_midi.sid_frequency(pitch), registers[0] | (registers[1] << 8))

    def test_guiro_and_vibraslap_are_noise_not_default_decay(self):
        for pitch in (58, 73, 74):
            registers = sid_midi._drum_registers(sid_midi.Note(0, 2, pitch, 100, 9, 0), 0, sid_midi.PAL_SID_CLOCK)
            self.assertEqual(registers[4] & 0xf0, sid_midi.NOISE)
            self.assertEqual(registers[5], 4)

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

    def test_channel_report_summarises_notes_and_families(self):
        notes = [
            sid_midi.Note(0, 100, 60, 100, 0, 80),
            sid_midi.Note(0, 50, 36, 90, 9, 0),
        ]
        rows = {row["channel"]: row for row in sid_midi.channel_report(notes)}
        self.assertEqual(rows[1]["notes"], 1)
        self.assertEqual(rows[1]["families"], ["lead"])
        self.assertEqual(rows[10]["families"], ["drums"])

    def test_exclude_channels_drops_only_those_channels(self):
        notes = [
            sid_midi.Note(0, 100, 60, 100, 0, 80),
            sid_midi.Note(0, 100, 36, 90, 9, 0),
        ]
        kept = sid_midi._filter_channels(notes, {9})
        self.assertEqual([note.channel for note in kept], [0])

    def test_exclude_channels_rejecting_every_note_is_an_error(self):
        notes = [sid_midi.Note(0, 100, 60, 100, 0, 80)]
        with self.assertRaises(ValueError):
            sid_midi._filter_channels(notes, {0})

    def test_trim_seconds_clips_notes_at_the_cutoff(self):
        division = 96
        tempo = 500_000  # 120 BPM: one quarter note per 0.5s
        notes = [
            sid_midi.Note(0, division, 60, 100, 0, 0),
            sid_midi.Note(division * 4, division * 5, 64, 100, 0, 0),
        ]
        trimmed = sid_midi._trim_notes(notes, 1.0, division, [(0, tempo)])
        self.assertEqual(len(trimmed), 1)
        self.assertEqual(trimmed[0].pitch, 60)

    def test_compile_sid_frames_honours_exclude_and_trim(self):
        header = b"MThd" + (6).to_bytes(4, "big") + b"\x00\x01\x00\x02\x00\x60"
        lead = midi_track((b"\x00\x90\x3c\x64", b"\x81\x60\x80\x3c\x00"))
        drums = midi_track((b"\x00\x99\x24\x64", b"\x60\x89\x24\x00"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "song.mid"
            path.write_bytes(header + lead + drums)
            full = sid_midi.compile_sid_frames(path)
            without_drums = sid_midi.compile_sid_frames(path, exclude_channels={9})
            trimmed = sid_midi.compile_sid_frames(path, trim_seconds=0.05)
        self.assertGreater(len(full), len(trimmed))
        self.assertTrue(any(frame[14:21] != bytes(7) for frame in full))
        self.assertTrue(all(frame[14:21] == bytes(7) for frame in without_drums))


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
        self.assertIn(bytes(build_prg.screen_code(ch) for ch in "BASS/DRUM"), data)
        # The oscilloscope is real bitmap pixels now (a shared, phase-cycling
        # picture per waveform shape, MIDI2AY-style), not PETSCII characters.
        for _, row_fn in build_prg.WAVEFORMS:
            self.assertIn(build_prg._wave_phase_bytes(row_fn, 0), data)
        # VIC-II hi-res bitmap mode gets switched on: LDA #$3B; STA $D011
        # (BMM|DEN|RSEL) followed by LDA #$18; STA $D018 (bitmap $2000 /
        # screen $0400).
        self.assertIn(bytes((0xa9, 0x3b, 0x8d, 0x11, 0xd0)), data)
        self.assertIn(bytes((0xa9, 0x18, 0x8d, 0x18, 0xd0)), data)

    def test_bad_frame_size_is_rejected(self):
        with self.assertRaises(ValueError):
            build_prg.encode_events([bytes(24)])


if __name__ == "__main__":
    unittest.main()
