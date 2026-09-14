from pathlib import Path

sid_path = Path("sid_midi.py")
test_path = Path("test_midi2sid.py")
sid = sid_path.read_text()
tests = test_path.read_text()

old = '''def _choose_voices(\n    tones: list[Note], previous_lead: tuple[int, int] | int | None = None,\n    previous_middle: tuple[int, int] | int | None = None,\n    lead_channel: int | None = None,\n) -> tuple[list[Note | None], tuple[int, int] | None, tuple[int, int] | None]:'''
new = '''def _choose_voices(\n    tones: list[Note], previous_lead: tuple[int, int] | int | None = None,\n    previous_middle: tuple[int, int] | int | None = None,\n    lead_channel: int | None = None, reserve_lead: bool = False,\n) -> tuple[list[Note | None], tuple[int, int] | None, tuple[int, int] | None]:'''
assert old in sid
sid = sid.replace(old, new)

old = '''    lead = max(preferred or pool, key=lead_score)\n    lead_key = lead.channel, lead.pitch\n\n    # Prefer an actual GM bass channel for voice three. Falling back to the\n    # lowest remaining pitch keeps the old two-note lead/bass behaviour.\n    remaining = [\n        note for note in pool\n        if (note.channel, note.pitch) != lead_key and note.pitch != lead.pitch\n    ]'''
new = '''    # Once the selected melody has started, reserve SID voice one for it.\n    # Letting a short arpeggio/effect jump into the gaps chops off the lead's\n    # SID release and makes the accompaniment appear to fight the tune. Intro\n    # and outro material can still use voice one outside the melody window.\n    if preferred:\n        lead = max(preferred, key=lead_score)\n    elif reserve_lead and lead_channel is not None:\n        lead = None\n    else:\n        lead = max(pool, key=lead_score)\n    lead_key = (lead.channel, lead.pitch) if lead is not None else None\n\n    # Prefer an actual GM bass channel for voice three. Falling back to the\n    # lowest remaining pitch keeps the old two-note lead/bass behaviour.\n    remaining = [\n        note for note in pool\n        if lead is None or (\n            (note.channel, note.pitch) != lead_key and note.pitch != lead.pitch\n        )\n    ]'''
assert old in sid
sid = sid.replace(old, new)

old = '''            + note.velocity\n            + continuity\n            - abs(note.pitch - lead.pitch)\n        )'''
new = '''            + note.velocity\n            + continuity\n            - (abs(note.pitch - lead.pitch) if lead is not None else 0)\n        )'''
assert old in sid
sid = sid.replace(old, new)

old = '''        attack = 0\n        release = min(patch.release, 2)\n        velocity_scale = 0.65 + 0.35 * note.velocity / 127'''
new = '''        attack = 0\n        # Voice one benefits from one extra SID release step. Previously every\n        # tight-mode voice was capped at 2, which made staccato melody patches\n        # such as Popcorn's pipe lead end more abruptly than their patch asks.\n        release = min(patch.release, 3 if voice == 0 else 2)\n        velocity_scale = 0.65 + 0.35 * note.velocity / 127'''
assert old in sid
sid = sid.replace(old, new)

old = '''    previous_lead = previous_middle = None\n    previous_sources: list[tuple | None] = [None, None, None]\n    lead_channel = _select_lead_channel(notes)\n\n    for frame in range(total):'''
new = '''    previous_lead = previous_middle = None\n    previous_sources: list[tuple | None] = [None, None, None]\n    lead_channel = _select_lead_channel(notes)\n    lead_notes = [note for note in notes if note.channel == lead_channel]\n    lead_start = min((note.start for note in lead_notes), default=None)\n    lead_end = max((note.end for note in lead_notes), default=None)\n    # Keep the last melodic oscillator/ADSR setup so a note-off frame can clear\n    # GATE without also destroying waveform and release. The SID envelope then\n    # gets to perform the release phase naturally until the next note arrives.\n    last_tone_registers: list[list[int] | None] = [None, None, None]\n\n    for frame in range(total):'''
assert old in sid
sid = sid.replace(old, new)

old = '''        voices, previous_lead, previous_middle = _choose_voices(\n            live_tones, previous_lead, previous_middle, lead_channel=lead_channel\n        )'''
new = '''        reserve_lead = (\n            lead_start is not None and lead_end is not None\n            and end_tick_scaled > lead_start * tick_denominator\n            and start_tick_scaled < lead_end * tick_denominator\n        )\n        voices, previous_lead, previous_middle = _choose_voices(\n            live_tones, previous_lead, previous_middle,\n            lead_channel=lead_channel, reserve_lead=reserve_lead,\n        )'''
assert old in sid
sid = sid.replace(old, new)

old = '''        registers = [0] * 25\n        for voice, note in enumerate(voices):\n            if note is None:\n                continue\n            onset_frame = note.start * tick_denominator // tick_numerator\n            age = max(0, frame - onset_frame)\n            registers[voice * 7:voice * 7 + 7] = _patch_registers(\n                note, age, voice, clock, feel=feel\n            )\n\n        sources: list[tuple | None] = ['''
new = '''        registers = [0] * 25\n        for voice, note in enumerate(voices):\n            base = voice * 7\n            if note is None:\n                previous = last_tone_registers[voice]\n                if previous is not None:\n                    release_registers = list(previous)\n                    release_registers[4] &= ~GATE\n                    registers[base:base + 7] = release_registers\n                continue\n            onset_frame = note.start * tick_denominator // tick_numerator\n            age = max(0, frame - onset_frame)\n            tone_registers = _patch_registers(note, age, voice, clock, feel=feel)\n            registers[base:base + 7] = tone_registers\n            last_tone_registers[voice] = tone_registers\n\n        sources: list[tuple | None] = ['''
assert old in sid
sid = sid.replace(old, new)

insert_after = '''    def test_selected_melody_channel_beats_short_arpeggio(self):\n        notes = [\n            sid_midi.Note(0, 120, 84, 112, 5, 79),\n            sid_midi.Note(0, 12, 96, 127, 2, 102),\n            sid_midi.Note(0, 120, 40, 104, 0, 35),\n        ]\n        voices, _, _ = sid_midi._choose_voices(notes, lead_channel=5)\n        self.assertEqual(voices[0].channel, 5)\n        self.assertEqual(voices[2].channel, 0)\n        self.assertIsNone(voices[1])\n\n'''
addition = '''    def test_reserved_lead_voice_rejects_arpeggio_during_melody_gap(self):\n        notes = [\n            sid_midi.Note(120, 132, 96, 127, 2, 102),\n            sid_midi.Note(100, 180, 40, 104, 0, 35),\n        ]\n        voices, _, _ = sid_midi._choose_voices(\n            notes, lead_channel=5, reserve_lead=True\n        )\n        self.assertIsNone(voices[0])\n        self.assertEqual(voices[2].channel, 0)\n\n    def test_lead_voice_fallback_remains_available_outside_melody_window(self):\n        notes = [\n            sid_midi.Note(0, 120, 72, 100, 6, 17),\n            sid_midi.Note(0, 120, 40, 100, 0, 35),\n        ]\n        voices, _, _ = sid_midi._choose_voices(\n            notes, lead_channel=5, reserve_lead=False\n        )\n        self.assertIsNotNone(voices[0])\n\n'''
assert insert_after in tests
tests = tests.replace(insert_after, insert_after + addition)

insert_after = '''    def test_tight_feel_uses_fast_attack_and_velocity_floor(self):\n        note = sid_midi.Note(0, 96, 60, 1, 0, 48)\n        tight = sid_midi.frames_for([note], 96, feel="tight")[0]\n        expressive = sid_midi.frames_for([note], 96, feel="expressive")[0]\n        self.assertEqual(tight[5] >> 4, 0)\n        self.assertEqual(expressive[5] >> 4, sid_midi.PATCHES["ensemble"].attack)\n        self.assertGreater(tight[6] >> 4, expressive[6] >> 4)\n\n'''
addition = '''    def test_note_off_preserves_waveform_and_sid_release(self):\n        note = sid_midi.Note(0, 12, 83, 112, 5, 79)\n        frames = sid_midi.frames_for([note], 96, feel="tight")\n        gated = [index for index, frame in enumerate(frames) if frame[4] & sid_midi.GATE]\n        self.assertTrue(gated)\n        release_frame = frames[gated[-1] + 1]\n        self.assertEqual(release_frame[4] & sid_midi.GATE, 0)\n        self.assertTrue(release_frame[4] & sid_midi.PULSE)\n        self.assertEqual(release_frame[6] & 0x0f, 3)\n\n'''
assert insert_after in tests
tests = tests.replace(insert_after, insert_after + addition)

sid_path.write_text(sid)
test_path.write_text(tests)
print("Applied lead ownership and SID release fixes")
