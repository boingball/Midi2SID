#!/usr/bin/env python3
"""Turn Standard MIDI notes into 50 Hz MOS SID register frames."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import math

FRAME_RATE = 50
PAL_SID_CLOCK = 985_248
NTSC_SID_CLOCK = 1_022_727

GATE = 0x01
SYNC = 0x02
RING = 0x04
TRIANGLE = 0x10
SAW = 0x20
PULSE = 0x40
NOISE = 0x80


@dataclass(frozen=True)
class Note:
    start: int
    end: int
    pitch: int
    velocity: int
    channel: int
    program: int


@dataclass(frozen=True)
class Patch:
    waveform: int
    attack: int
    decay: int
    sustain: int
    release: int
    pulse: int = 0x800
    vibrato: int = 0
    vibrato_delay: int = 8
    vibrato_step: int = 3
    pwm_depth: int = 0
    pwm_step: int = 4
    filter_cutoff: int = 0
    resonance: int = 0
    filter_mode: int = 0x10
    ring: bool = False
    sync: bool = False


PATCHES = {
    "piano":      Patch(PULSE, 0, 8, 5, 4, 0x680, pwm_depth=0x080),
    "chromatic":  Patch(TRIANGLE, 0, 4, 3, 8, ring=True),
    "organ":      Patch(PULSE, 2, 3, 12, 5, 0x500, vibrato=5, pwm_depth=0x180),
    "guitar":     Patch(SAW, 0, 7, 4, 4, vibrato=4),
    "bass":       Patch(PULSE, 0, 5, 10, 4, 0x350, pwm_depth=0x100, filter_cutoff=580, resonance=8),
    "strings":    Patch(SAW, 6, 5, 11, 8, vibrato=8, pwm_depth=0x100, filter_cutoff=1150, resonance=5),
    "ensemble":   Patch(PULSE, 7, 4, 10, 8, 0x700, vibrato=10, pwm_depth=0x240, filter_cutoff=1250, resonance=6),
    "brass":      Patch(SAW, 1, 4, 11, 5, filter_cutoff=950, resonance=10),
    "reed":       Patch(PULSE, 2, 5, 10, 5, 0x420, vibrato=9, pwm_depth=0x100, filter_cutoff=1450, resonance=4),
    "pipe":       Patch(TRIANGLE, 2, 3, 12, 5, vibrato=11),
    "lead":       Patch(PULSE | SAW, 0, 3, 12, 4, 0x800, vibrato=12, pwm_depth=0x180, filter_cutoff=1500, resonance=7, sync=True),
    "pad":        Patch(PULSE, 9, 4, 9, 10, 0x900, vibrato=7, pwm_depth=0x300, filter_cutoff=850, resonance=8),
    "effects":    Patch(SAW, 1, 7, 7, 8, vibrato=28, filter_cutoff=1200, resonance=12, ring=True),
    "ethnic":     Patch(TRIANGLE, 0, 7, 5, 5, vibrato=6, ring=True),
    "percussive": Patch(NOISE, 0, 5, 2, 3),
    "sfx":        Patch(PULSE | SAW, 0, 5, 7, 8, 0x800, vibrato=35, pwm_depth=0x300, filter_cutoff=1000, resonance=13, sync=True),
}

LFO = (0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5)


def vlq(data: bytes, position: int) -> tuple[int, int]:
    value = 0
    while True:
        if position >= len(data):
            raise ValueError("truncated MIDI variable-length quantity")
        byte = data[position]
        position += 1
        value = (value << 7) | (byte & 0x7f)
        if byte < 0x80:
            return value, position


def read_midi(path: str | Path) -> tuple[int, list[Note], list[tuple[int, int]]]:
    """Read format 0/1 PPQN MIDI without external Python packages."""
    data = Path(path).read_bytes()
    if data[:4] != b"MThd" or len(data) < 14:
        raise ValueError("not a Standard MIDI file")
    header_size = int.from_bytes(data[4:8], "big")
    midi_format = int.from_bytes(data[8:10], "big")
    track_count = int.from_bytes(data[10:12], "big")
    division = int.from_bytes(data[12:14], "big")
    if midi_format not in (0, 1) or division & 0x8000:
        raise ValueError("only format 0/1 PPQN MIDI is supported")

    position = 8 + header_size
    notes: list[Note] = []
    tempos: list[tuple[int, int]] = []
    for _ in range(track_count):
        if data[position:position + 4] != b"MTrk":
            raise ValueError("invalid MIDI track")
        size = int.from_bytes(data[position + 4:position + 8], "big")
        track = data[position + 8:position + 8 + size]
        position += 8 + size
        cursor = tick = 0
        running = None
        active: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
        programs = [0] * 16
        while cursor < len(track):
            delta, cursor = vlq(track, cursor)
            tick += delta
            status = track[cursor]
            if status < 0x80:
                if running is None:
                    raise ValueError("invalid MIDI running status")
                status = running
            else:
                cursor += 1
                if status < 0xf0:
                    running = status
            kind, channel = status & 0xf0, status & 0x0f
            if status == 0xff:
                meta = track[cursor]
                cursor += 1
                length, cursor = vlq(track, cursor)
                payload = track[cursor:cursor + length]
                cursor += length
                if meta == 0x51 and len(payload) == 3:
                    tempos.append((tick, int.from_bytes(payload, "big")))
                continue
            if status in (0xf0, 0xf7):
                length, cursor = vlq(track, cursor)
                cursor += length
                continue
            length = 1 if kind in (0xc0, 0xd0) else 2
            payload = track[cursor:cursor + length]
            cursor += length
            if kind == 0xc0:
                programs[channel] = payload[0]
            elif kind == 0x90 and payload[1]:
                active[(channel, payload[0])].append((tick, payload[1], programs[channel]))
            elif kind in (0x80, 0x90):
                key = channel, payload[0]
                if active[key]:
                    start, velocity, program = active[key].pop(0)
                    if tick > start:
                        notes.append(Note(start, tick, payload[0], velocity, channel, program))
    if not notes:
        raise ValueError("MIDI contains no closed note events")
    return division, notes, sorted(set(tempos))


def family(program: int) -> str:
    names = (
        "piano", "chromatic", "organ", "guitar", "bass", "strings",
        "ensemble", "brass", "reed", "pipe", "lead", "pad", "effects",
        "ethnic", "percussive", "sfx",
    )
    return names[max(0, min(127, program)) // 8]


def sid_frequency(pitch: float, clock: int = PAL_SID_CLOCK) -> int:
    hz = 440.0 * 2 ** ((pitch - 69.0) / 12.0)
    return max(1, min(0xffff, round(hz * (1 << 24) / clock)))


def _choose_voices(
    tones: list[Note], previous_lead: int | None = None,
    previous_middle: int | None = None,
) -> tuple[list[Note | None], int | None, int | None]:
    """Choose lead, accompaniment and bass while preserving melodic continuity."""
    by_pitch: dict[int, Note] = {}
    for note in tones:
        old = by_pitch.get(note.pitch)
        if old is None or note.velocity > old.velocity:
            by_pitch[note.pitch] = note
    pitches = sorted(by_pitch)
    if not pitches:
        return [None, None, None], None, None
    bass_pitch = pitches[0]
    candidates = pitches[1:] or pitches

    def lead_score(pitch: int) -> int:
        note = by_pitch[pitch]
        bonus = {
            "lead": 500, "brass": 220, "reed": 180, "pipe": 150,
            "strings": 100, "pad": -180, "bass": -100,
        }.get(family(note.program), 0)
        short = 2 * max(0, 300 - min(300, note.end - note.start))
        continuity = -3 * abs(pitch - previous_lead) if previous_lead is not None else 0
        return bonus + short + note.velocity + continuity

    lead_pitch = max(candidates, key=lead_score)
    middle_pitches = [pitch for pitch in pitches if pitch not in {bass_pitch, lead_pitch}]
    if previous_middle in middle_pitches:
        middle_pitch = previous_middle
    elif middle_pitches:
        middle_pitch = middle_pitches[-1]
    else:
        middle_pitch = lead_pitch
    return [by_pitch[lead_pitch], by_pitch[middle_pitch], by_pitch[bass_pitch]], lead_pitch, middle_pitch


def _patch_registers(note: Note, age: int, voice: int, clock: int) -> list[int]:
    patch = PATCHES[family(note.program)]
    cents = 0.0
    if patch.vibrato and age >= patch.vibrato_delay:
        cents = patch.vibrato * LFO[((age - patch.vibrato_delay) // patch.vibrato_step) % len(LFO)]
    frequency = sid_frequency(note.pitch + cents / 100.0, clock)
    pulse = patch.pulse
    if patch.pwm_depth:
        pulse += round(patch.pwm_depth * LFO[(age // patch.pwm_step) % len(LFO)])
    pulse = max(0x080, min(0xf80, pulse))
    velocity_sustain = max(1, min(15, round(patch.sustain * note.velocity / 127)))
    control = patch.waveform | GATE
    if patch.ring and voice:
        control |= RING
    if patch.sync and voice:
        control |= SYNC
    return [
        frequency & 0xff, frequency >> 8,
        pulse & 0xff, (pulse >> 8) & 0x0f,
        control,
        (patch.attack << 4) | patch.decay,
        (velocity_sustain << 4) | patch.release,
    ]


DRUM_PRIORITY = {
    38: 100, 40: 98, 39: 96, 35: 94, 36: 94, 54: 90,
    42: 88, 44: 87, 46: 86, 49: 82, 57: 82,
}


def _drum_registers(note: Note, age: int, clock: int) -> list[int]:
    pitch = note.pitch
    if pitch in (35, 36):
        waveform, musical_pitch, decay = TRIANGLE, 36 - min(age, 3) * 5, 5
    elif pitch in (41, 43, 45, 47, 48, 50):
        waveform, musical_pitch, decay = PULSE, pitch - 12 - min(age, 2) * 2, 6
    elif pitch in (56, 75, 76, 77, 80, 81):
        waveform, musical_pitch, decay = PULSE, pitch, 5
    else:
        waveform, musical_pitch = NOISE, max(24, min(96, pitch + 18))
        decay = 2 if pitch in (42, 44, 54, 69, 70) else 6
    frequency = sid_frequency(musical_pitch, clock)
    sustain = max(1, min(15, round(15 * note.velocity / 127)))
    return [
        frequency & 0xff, frequency >> 8, 0, 8,
        waveform | GATE,
        decay,
        (sustain << 4) | 2,
    ]


def _first_tempo(tempos: list[tuple[int, int]]) -> int:
    at_zero = [value for tick, value in tempos if tick == 0]
    return at_zero[-1] if at_zero else (tempos[0][1] if tempos else 500_000)


def frames_for(
    notes: list[Note], division: int, tempos: list[tuple[int, int]] | None = None,
    drums: str = "smart", video: str = "pal",
) -> list[bytes]:
    """Render complete 25-register SID snapshots at the target video rate."""
    if drums not in ("off", "smart"):
        raise ValueError("drums must be off or smart")
    if video not in ("pal", "ntsc"):
        raise ValueError("video must be pal or ntsc")
    tempo = _first_tempo(tempos or [])
    rate = 50 if video == "pal" else 60
    clock = PAL_SID_CLOCK if video == "pal" else NTSC_SID_CLOCK
    step = max(1, round(division * 1_000_000 / (tempo * rate)))
    total = max(1, math.ceil((max(note.end for note in notes) + step) / step))
    frames: list[bytes] = []
    previous_lead = previous_middle = None

    for frame in range(total):
        start, end = frame * step, (frame + 1) * step
        live_tones = [n for n in notes if n.channel != 9 and n.start < end and n.end > start]
        voices, previous_lead, previous_middle = _choose_voices(
            live_tones, previous_lead, previous_middle
        )
        if not live_tones:
            previous_lead = previous_middle = None
        registers = [0] * 25
        for voice, note in enumerate(voices):
            if note is None:
                continue
            age = max(0, (start - note.start) // step)
            registers[voice * 7:voice * 7 + 7] = _patch_registers(note, age, voice, clock)

        if drums != "off":
            candidates = []
            for note in notes:
                if note.channel != 9:
                    continue
                onset = note.start // step
                age = frame - onset
                tail = 2 if note.pitch in (42, 44, 54, 69, 70) else 6
                if 0 <= age < tail:
                    candidates.append((note, age))
            if candidates:
                drum, age = max(candidates, key=lambda item: (DRUM_PRIORITY.get(item[0].pitch, 50), item[0].velocity))
                registers[14:21] = _drum_registers(drum, age, clock)

        # Use the lead patch to drive the shared filter and route voice one.
        lead = voices[0]
        if lead is not None:
            patch = PATCHES[family(lead.program)]
            if patch.filter_cutoff:
                cutoff = patch.filter_cutoff
                registers[21] = cutoff & 7
                registers[22] = (cutoff >> 3) & 0xff
                registers[23] = (patch.resonance << 4) | 1
                registers[24] = patch.filter_mode | 15
            else:
                registers[24] = 15
        else:
            registers[24] = 15
        frames.append(bytes(registers))
    return frames


def compile_sid_frames(
    midi_path: str | Path, video: str = "pal", drums: str = "smart",
) -> list[bytes]:
    division, notes, tempos = read_midi(midi_path)
    return frames_for(notes, division, tempos, drums=drums, video=video)
