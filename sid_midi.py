#!/usr/bin/env python3
"""Turn Standard MIDI notes into 50 Hz MOS SID register frames."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
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


class SidFrame(bytes):
    """A SID register snapshot plus voices that need an ADSR retrigger."""

    def __new__(cls, registers: bytes | bytearray | list[int], retrigger_mask: int = 0):
        frame = super().__new__(cls, bytes(registers))
        frame.retrigger_mask = retrigger_mask & 0x07
        return frame


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
    # GM pipe/ocarina parts are often the actual tune in downloadable MIDIs.
    # Triangle was too quiet beside two pulse voices, so use a bright, nearly
    # square pulse that reads as a classic SID lead without harsh sync.
    "pipe":       Patch(PULSE, 0, 3, 13, 3, 0x780, vibrato=8, pwm_depth=0x080),
    "lead":       Patch(PULSE, 0, 3, 12, 4, 0x800, vibrato=12, pwm_depth=0x180, filter_cutoff=1500, resonance=7, sync=True),
    "pad":        Patch(PULSE, 9, 4, 9, 10, 0x900, vibrato=7, pwm_depth=0x300, filter_cutoff=850, resonance=8),
    "effects":    Patch(SAW, 1, 7, 7, 8, vibrato=28, filter_cutoff=1200, resonance=12, ring=True),
    "ethnic":     Patch(TRIANGLE, 0, 7, 5, 5, vibrato=6, ring=True),
    "percussive": Patch(NOISE, 0, 5, 2, 3),
    "sfx":        Patch(SAW, 0, 5, 7, 8, 0x800, vibrato=35, pwm_depth=0x300, filter_cutoff=1000, resonance=13, sync=True),
}

LFO = (0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5)

LEAD_FAMILY_SCORE = {
    "lead": 1000, "pipe": 850, "reed": 700, "brass": 550,
    "ensemble": 350, "strings": 320, "organ": 250, "guitar": 120,
    "piano": 80, "chromatic": 80, "pad": 100, "ethnic": 100,
    "effects": -300, "bass": -1000, "percussive": -300, "sfx": -400,
}

MIDDLE_FAMILY_SCORE = {
    "ensemble": 500, "strings": 460, "pad": 420, "organ": 360,
    "brass": 220, "reed": 180, "guitar": 140, "piano": 120,
    "chromatic": 100, "pipe": 80, "lead": 80, "ethnic": 60,
    "effects": -300, "bass": -450, "percussive": -500, "sfx": -450,
}


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


def _coverage(notes: list[Note]) -> int:
    """Return channel activity in ticks with overlapping notes counted once."""
    intervals = sorted((note.start, note.end) for note in notes)
    if not intervals:
        return 0
    start, end = intervals[0]
    total = 0
    for next_start, next_end in intervals[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def _max_polyphony(notes: list[Note]) -> int:
    events: list[tuple[int, int]] = []
    for note in notes:
        events.extend(((note.start, 1), (note.end, -1)))
    active = maximum = 0
    for _, change in sorted(events, key=lambda item: (item[0], item[1])):
        active += change
        maximum = max(maximum, active)
    return maximum


def _select_lead_channel(notes: list[Note]) -> int | None:
    """Identify the song's melodic channel before reducing it frame by frame.

    Re-electing the lead from every simultaneous chord rewards short arpeggios
    and makes a recognisable melody jump between SID oscillators. A whole-song
    channel score gives voice one a stable owner while still allowing fallback
    material to play during rests in that channel.
    """
    channels: dict[int, list[Note]] = defaultdict(list)
    for note in notes:
        if note.channel != 9:
            channels[note.channel].append(note)
    if not channels:
        return None
    song_end = max(note.end for channel_notes in channels.values() for note in channel_notes)
    scored: list[tuple[float, float, int, int]] = []
    for channel, channel_notes in channels.items():
        family_counts: dict[str, int] = defaultdict(int)
        for note in channel_notes:
            family_counts[family(note.program)] += 1
        dominant_family = max(family_counts, key=family_counts.get)
        count = len(channel_notes)
        average_pitch = sum(note.pitch for note in channel_notes) / count
        average_velocity = sum(note.velocity for note in channel_notes) / count
        activity = _coverage(channel_notes) / max(1, song_end)
        polyphony_penalty = max(0, _max_polyphony(channel_notes) - 1) * 100
        score = (
            LEAD_FAMILY_SCORE.get(dominant_family, 0)
            + min(count, 256) * 2
            + average_pitch * 4
            + average_velocity * 2
            + min(1.0, activity) * 300
            - polyphony_penalty
        )
        scored.append((score, average_pitch, count, channel))
    return max(scored)[3]


# Families strong enough to plausibly carry a song's melody on their own,
# reusing _select_lead_channel's own score table (lead/pipe/reed/brass all
# score >=500 there). Restricting the secondary lead pick to these keeps
# heavily-scored backing parts (ensembles, pads, organs, rhythm guitars) from
# ever qualifying just because they are busy or high-pitched.
_STRONG_LEAD_FAMILIES = frozenset(
    family_name for family_name, score in LEAD_FAMILY_SCORE.items() if score >= 500
)


def _select_secondary_lead_channel(notes: list[Note], primary: int | None) -> int | None:
    """Find a second channel that also plausibly carries the melody elsewhere.

    A full-band arrangement often gives an intro or bridge its own solo
    voice on a different patch than the song's main lead (e.g. a brass-patch
    riff before a synth lead takes over): reserving only the single top-
    scoring channel left voice one with no good candidate during that
    section, falling back to whichever chord/pad note briefly scored highest
    that frame and sounding like backing material fighting for the lead.
    Restricted to the same strong "solo" GM families as the primary channel
    and to a real runner-up score, so a busy accompaniment part never
    qualifies just from note count or pitch.
    """
    channels: dict[int, list[Note]] = defaultdict(list)
    for note in notes:
        if note.channel != 9 and note.channel != primary:
            channels[note.channel].append(note)
    if not channels:
        return None
    song_end = max(note.end for channel_notes in channels.values() for note in channel_notes)
    best_score = float("-inf")
    best_channel = None
    for channel, channel_notes in channels.items():
        family_counts: dict[str, int] = defaultdict(int)
        for note in channel_notes:
            family_counts[family(note.program)] += 1
        dominant_family = max(family_counts, key=family_counts.get)
        if dominant_family not in _STRONG_LEAD_FAMILIES:
            continue
        count = len(channel_notes)
        average_pitch = sum(note.pitch for note in channel_notes) / count
        average_velocity = sum(note.velocity for note in channel_notes) / count
        activity = _coverage(channel_notes) / max(1, song_end)
        polyphony_penalty = max(0, _max_polyphony(channel_notes) - 1) * 100
        score = (
            LEAD_FAMILY_SCORE.get(dominant_family, 0)
            + min(count, 256) * 2
            + average_pitch * 4
            + average_velocity * 2
            + min(1.0, activity) * 300
            - polyphony_penalty
        )
        if score > best_score:
            best_score, best_channel = score, channel
    return best_channel if best_score >= 500 else None


def _merged_lead_intervals(channel_notes: list[Note], division: int) -> list[tuple[int, int]]:
    """Merge a channel's own notes into (start, end) spans, bridging rests of
    a bar (4 beats) or less. Reserving voice one only within these spans
    blocks a busy arpeggio from stealing a brief breath without silencing
    voice one across a whole section where the channel genuinely isn't
    playing (see frames_for).
    """
    intervals: list[tuple[int, int]] = []
    if not channel_notes:
        return intervals
    ordered = sorted(channel_notes, key=lambda note: note.start)
    max_gap = max(1, division * 4)
    interval_start = ordered[0].start
    interval_end = ordered[0].end
    for note in ordered[1:]:
        if note.start - interval_end > max_gap:
            intervals.append((interval_start, interval_end))
            interval_start = note.start
        interval_end = max(interval_end, note.end)
    intervals.append((interval_start, interval_end))
    return intervals


def _frame_in_intervals(
    intervals: list[tuple[int, int]], start_scaled: int, end_scaled: int, denominator: int,
) -> bool:
    return any(
        end_scaled > interval_start * denominator and start_scaled < interval_end * denominator
        for interval_start, interval_end in intervals
    )


def _choose_voices(
    tones: list[Note], previous_lead: tuple[int, int] | int | None = None,
    previous_middle: tuple[int, int] | int | None = None,
    lead_channel: int | None = None, reserve_lead: bool = False,
) -> tuple[list[Note | None], tuple[int, int] | None, tuple[int, int] | None]:
    """Choose stable lead, restrained accompaniment and bass SID voices."""
    unique: dict[tuple[int, int], Note] = {}
    for note in tones:
        key = note.channel, note.pitch
        old = unique.get(key)
        if old is None or (note.velocity, note.start) > (old.velocity, old.start):
            unique[key] = note
    pool = list(unique.values())
    if not pool:
        return [None, None, None], None, None

    preferred = [note for note in pool if lead_channel is not None and note.channel == lead_channel]

    def lead_score(note: Note) -> int:
        continuity = 250 if previous_lead == (note.channel, note.pitch) else 0
        return (
            {
                "lead": 650, "pipe": 550, "reed": 450, "brass": 350,
                "ensemble": 180, "strings": 160, "organ": 120,
                "pad": -100, "effects": -220, "bass": -600,
                "percussive": -400, "sfx": -350,
            }.get(family(note.program), 0)
            + note.pitch * 4
            + note.velocity * 2
            + min(note.end - note.start, 240)
            + continuity
        )

    # Once the selected melody has started, reserve SID voice one for it.
    # Letting a short arpeggio/effect jump into the gaps chops off the lead's
    # SID release and makes the accompaniment appear to fight the tune. Intro
    # and outro material can still use voice one outside the melody window.
    if preferred:
        lead = max(preferred, key=lead_score)
    elif reserve_lead and lead_channel is not None:
        lead = None
    else:
        lead = max(pool, key=lead_score)
    lead_key = (lead.channel, lead.pitch) if lead is not None else None

    # Prefer an actual GM bass channel for voice three. Falling back to the
    # lowest remaining pitch keeps the old two-note lead/bass behaviour.
    remaining = [
        note for note in pool
        if lead is None or (
            (note.channel, note.pitch) != lead_key and note.pitch != lead.pitch
        )
    ]
    bass_candidates = [note for note in remaining if family(note.program) == "bass"]
    bass_pool = bass_candidates or remaining
    bass = min(
        bass_pool,
        key=lambda note: (note.pitch, -(note.end - note.start), -note.velocity),
    ) if bass_pool else None

    accompaniment = [
        note for note in remaining
        if bass is None or (note.channel, note.pitch) != (bass.channel, bass.pitch)
    ]

    def middle_score(note: Note) -> int:
        continuity = 350 if previous_middle == (note.channel, note.pitch) else 0
        return (
            MIDDLE_FAMILY_SCORE.get(family(note.program), 0)
            + min(note.end - note.start, 480)
            + note.velocity
            + continuity
            - (abs(note.pitch - lead.pitch) if lead is not None else 0)
        )

    middle = max(accompaniment, key=middle_score) if accompaniment else None
    # A frantic effects/arpeggio note should not fill a SID voice merely because
    # one is free. Silence is cleaner than constant retrigger chatter.
    if middle is not None and middle_score(middle) < 180:
        middle = None

    middle_key = (middle.channel, middle.pitch) if middle is not None else None
    return [lead, middle, bass], lead_key, middle_key


def _patch_registers(
    note: Note, age: int, voice: int, clock: int, feel: str = "tight",
) -> list[int]:
    patch = PATCHES[family(note.program)]
    cents = 0.0
    if patch.vibrato and age >= patch.vibrato_delay:
        cents = patch.vibrato * LFO[((age - patch.vibrato_delay) // patch.vibrato_step) % len(LFO)]
    frequency = sid_frequency(note.pitch + cents / 100.0, clock)
    pulse = patch.pulse
    if patch.pwm_depth and feel == "expressive":
        pulse += round(patch.pwm_depth * LFO[(age // patch.pwm_step) % len(LFO)])
    pulse = max(0x080, min(0xf80, pulse))
    if feel == "tight":
        # Most automatically arranged notes last only a few video frames. Slow
        # SID attacks make them arrive perceptibly behind the MIDI beat, so the
        # default mode uses the chip's fastest attack and a short release while
        # retaining each patch's waveform, pulse colour, decay, sustain and
        # vibrato. Animated PWM remains available in expressive mode: restarting
        # it on every short note makes dense automatic reductions sound untidy.
        attack = 0
        # Voice one benefits from one extra SID release step. Previously every
        # tight-mode voice was capped at 2, which made staccato melody patches
        # such as Popcorn's pipe lead end more abruptly than their patch asks.
        release = min(patch.release, 3 if voice == 0 else 2)
        velocity_scale = 0.65 + 0.35 * note.velocity / 127
    else:
        attack = patch.attack
        release = patch.release
        velocity_scale = note.velocity / 127
    # SID has one master volume rather than a mixer per oscillator. Balance the
    # automatic reduction through ADSR sustain: voice one stays prominent while
    # backing and bass retain headroom instead of masking the melody.
    role_scale = (1.18, 0.78, 0.72)[voice]
    velocity_sustain = max(1, min(15, round(patch.sustain * velocity_scale * role_scale)))
    control = patch.waveform | GATE
    if patch.ring and voice:
        control |= RING
    if patch.sync and voice:
        control |= SYNC
    return [
        frequency & 0xff, frequency >> 8,
        pulse & 0xff, (pulse >> 8) & 0x0f,
        control,
        (attack << 4) | patch.decay,
        (velocity_sustain << 4) | release,
    ]


DRUM_PRIORITY = {
    38: 100, 40: 98, 39: 96, 35: 94, 36: 94, 54: 90,
    42: 88, 44: 87, 46: 86, 49: 82, 57: 82,
    37: 90, 53: 75, 60: 70, 61: 70, 62: 70, 63: 70, 64: 70,
    65: 70, 66: 70, 67: 65, 68: 65, 58: 60, 73: 60, 74: 60,
}

# Hand drums (bongo/conga/timbale) and cuica are pitched membranophones like
# the toms, just without their own General MIDI tom pitch spread, so they
# share the toms' "pulse tuned down an octave, sharpening as it decays" shape.
_HAND_DRUMS = (60, 61, 62, 63, 64, 65, 66, 78, 79)
# Metallic/bell percussion and the two whistles are steady pitched tones at
# their own GM note, the same treatment already used for cowbell/claves/
# woodblocks/triangle.
_BELL_TONES = (53, 56, 67, 68, 71, 72, 75, 76, 77, 80, 81)
# A scrape or rattle: longer than a hi-hat tick, shorter than a cymbal wash.
_SCRAPE_NOISE = (58, 73, 74)


def _drum_registers(note: Note, age: int, clock: int) -> list[int]:
    pitch = note.pitch
    if pitch in (35, 36):
        waveform, musical_pitch, decay = TRIANGLE, 36 - min(age, 3) * 5, 5
    elif pitch in (41, 43, 45, 47, 48, 50) or pitch in _HAND_DRUMS:
        waveform, musical_pitch, decay = PULSE, pitch - 12 - min(age, 2) * 2, 6
    elif pitch in _BELL_TONES:
        waveform, musical_pitch, decay = PULSE, pitch, 5
    elif pitch in _SCRAPE_NOISE:
        waveform, musical_pitch, decay = NOISE, max(24, min(96, pitch + 18)), 4
    else:
        waveform, musical_pitch = NOISE, max(24, min(96, pitch + 18))
        decay = 2 if pitch in (37, 42, 44, 54, 69, 70) else 6
    frequency = sid_frequency(musical_pitch, clock)
    # A high sustain level makes short noise hits stick at full volume until
    # GATE drops, which is heard as loud clicks or bursts of continuous noise.
    # SID percussion is a one-shot decay, so its sustain level is always zero.
    return [
        frequency & 0xff, frequency >> 8, 0, 8,
        waveform | GATE,
        decay,
        2,
    ]


def _drum_tail(pitch: int) -> int:
    """Return a short, bass-friendly percussion lifetime in video frames."""
    if pitch in (37, 42, 44, 46, 54, 69, 70):   # side stick, hats, tambourine, shakers
        return 1
    if pitch in (49, 51, 52, 55, 57, 59):       # cymbals
        return 3
    if pitch in (41, 43, 45, 47, 48, 50) or pitch in _HAND_DRUMS:  # toms, bongo/conga/timbale, cuica
        return 3
    return 2                                    # kick, snare, bells and other hits


def _first_tempo(tempos: list[tuple[int, int]]) -> int:
    at_zero = [value for tick, value in tempos if tick == 0]
    return at_zero[-1] if at_zero else (tempos[0][1] if tempos else 500_000)


def frames_for(
    notes: list[Note], division: int, tempos: list[tuple[int, int]] | None = None,
    drums: str = "smart", video: str = "pal", filter_mode: str = "off",
    feel: str = "tight",
) -> list[bytes]:
    """Render complete 25-register SID snapshots at the target video rate."""
    if drums not in ("off", "smart"):
        raise ValueError("drums must be off or smart")
    if video not in ("pal", "ntsc"):
        raise ValueError("video must be pal or ntsc")
    if filter_mode not in ("off", "auto"):
        raise ValueError("filter mode must be off or auto")
    if feel not in ("tight", "expressive"):
        raise ValueError("feel must be tight or expressive")
    tempo = _first_tempo(tempos or [])
    rate = 50 if video == "pal" else 60
    clock = PAL_SID_CLOCK if video == "pal" else NTSC_SID_CLOCK
    ticks_per_frame = Fraction(division * 1_000_000, tempo * rate)
    tick_numerator = ticks_per_frame.numerator
    tick_denominator = ticks_per_frame.denominator
    last_tick = max(note.end for note in notes)
    total = max(1, math.ceil(last_tick * tick_denominator / tick_numerator) + 1)
    frames: list[bytes] = []
    previous_lead = previous_middle = None
    previous_sources: list[tuple | None] = [None, None, None]
    lead_channel = _select_lead_channel(notes)
    lead_intervals = _merged_lead_intervals(
        [note for note in notes if note.channel == lead_channel], division
    )
    # A second channel that also carries real melodic material elsewhere
    # (e.g. an intro riff on a different patch before the main lead takes
    # over) gets the same reserved-span treatment, so voice one falls back
    # to it instead of whichever backing chord briefly scores highest.
    secondary_lead_channel = _select_secondary_lead_channel(notes, lead_channel)
    secondary_lead_intervals = _merged_lead_intervals(
        [note for note in notes if note.channel == secondary_lead_channel], division
    )
    # Keep the last melodic oscillator/ADSR setup so a note-off frame can clear
    # GATE without also destroying waveform and release. The SID envelope then
    # gets to perform the release phase naturally until the next note arrives.
    last_tone_registers: list[list[int] | None] = [None, None, None]

    for frame in range(total):
        # Keep the MIDI clock fractional. Rounding 5.28 ticks/frame to 5 made
        # the Popcorn test file 5.6% slow and lets the error accumulate over a
        # song. Integer cross-products avoid floating-point boundary wobble.
        start_tick_scaled = frame * tick_numerator
        end_tick_scaled = (frame + 1) * tick_numerator
        live_tones = [
            n for n in notes
            if n.channel != 9
            and n.start * tick_denominator < end_tick_scaled
            and n.end * tick_denominator > start_tick_scaled
        ]
        if _frame_in_intervals(lead_intervals, start_tick_scaled, end_tick_scaled, tick_denominator):
            active_lead_channel, reserve_lead = lead_channel, True
        elif _frame_in_intervals(secondary_lead_intervals, start_tick_scaled, end_tick_scaled, tick_denominator):
            active_lead_channel, reserve_lead = secondary_lead_channel, True
        else:
            active_lead_channel, reserve_lead = lead_channel, False
        voices, previous_lead, previous_middle = _choose_voices(
            live_tones, previous_lead, previous_middle,
            lead_channel=active_lead_channel, reserve_lead=reserve_lead,
        )
        if not live_tones:
            previous_lead = previous_middle = None
        registers = [0] * 25
        for voice, note in enumerate(voices):
            base = voice * 7
            if note is None:
                previous = last_tone_registers[voice]
                if previous is not None:
                    release_registers = list(previous)
                    release_registers[4] &= ~GATE
                    registers[base:base + 7] = release_registers
                continue
            onset_frame = note.start * tick_denominator // tick_numerator
            age = max(0, frame - onset_frame)
            tone_registers = _patch_registers(note, age, voice, clock, feel=feel)
            registers[base:base + 7] = tone_registers
            last_tone_registers[voice] = tone_registers

        sources: list[tuple | None] = [
            ("tone", note.start, note.end, note.pitch, note.channel, note.program)
            if note is not None else None
            for note in voices
        ]

        if drums != "off":
            candidates = []
            for note in notes:
                if note.channel != 9:
                    continue
                onset = note.start * tick_denominator // tick_numerator
                age = frame - onset
                tail = _drum_tail(note.pitch)
                if 0 <= age < tail:
                    candidates.append((note, age))
            if candidates:
                drum, age = max(candidates, key=lambda item: (DRUM_PRIORITY.get(item[0].pitch, 50), item[0].velocity))
                registers[14:21] = _drum_registers(drum, age, clock)
                sources[2] = ("drum", drum.start, drum.pitch, drum.channel)

        # Use the lead patch to drive the shared filter and route voice one.
        lead = voices[0]
        if filter_mode == "auto" and lead is not None:
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
        retrigger_mask = 0
        for voice, source in enumerate(sources):
            if source is not None and previous_sources[voice] is not None and source != previous_sources[voice]:
                retrigger_mask |= 1 << voice
        frames.append(SidFrame(registers, retrigger_mask))
        previous_sources = sources
    return frames


def compile_sid_frames(
    midi_path: str | Path, video: str = "pal", drums: str = "smart",
    filter_mode: str = "off", feel: str = "tight",
) -> list[bytes]:
    division, notes, tempos = read_midi(midi_path)
    return frames_for(
        notes, division, tempos, drums=drums, video=video,
        filter_mode=filter_mode, feel=feel,
    )
