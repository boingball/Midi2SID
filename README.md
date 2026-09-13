# MIDI2SID

Smart MIDI-to-Commodore 64 SID conversion, inspired by the voice-routing work
in [MIDI2AY](https://github.com/boingball/Midi2AY).

The first version reads format 0/1 PPQN Standard MIDI files and creates a
self-running C64 `.prg`. It automatically reduces polyphonic arrangements to
the SID's three voices, maps all 128 General MIDI programs to SID synth
families, and gives MIDI channel 10 a dedicated synthesized drum path.

## Quick start

Python 3.10 or newer is required. The converter itself has no dependencies.

```sh
python3 midi2sid.py song.mid song.prg
```

Useful options:

```sh
python3 midi2sid.py song.mid song.prg --title "MY SONG" --drums smart --video pal
python3 midi2sid.py song.mid song.prg --drums off --video ntsc
python3 midi2sid.py song.mid song.prg --filter auto
```

Load the resulting program in VICE, another C64 emulator, a flash cartridge,
or real hardware, then type `RUN`. The program supplies its own BASIC launcher,
50/60 Hz machine-code player and generated text-mode title screen.

## What the automatic patches use

- triangle, sawtooth, pulse and noise waveforms
- SID ADSR envelopes
- velocity-scaled sustain
- pulse-width modulation
- vibrato, ring modulation and oscillator sync
- shared resonant filter routing
- tonal kick/tom sweeps and noise percussion

The shared filter is off by default because rapid 6581 filter-route changes can
click and filter calibration varies between chips. `--filter auto` enables the
more aggressive patch filter treatment.

The generated player starts in safe visual mode 1 with a black border. During
playback, keys 1-5 select: safe/static, voice lights, slow border, both, and a
slow demo colour mode. No mode changes colour faster than about 3 Hz.

This is an automatic chip-music arrangement, not a transparent reproduction of
the source MIDI. Dense chords must be reduced to three voices, and drums borrow
voice three while they sound.

## Current limits

- The PRG v1 player combines register deltas with a streaming, 256-byte-window
  LZSS decoder. Very long or modulation-heavy songs can still exceed `$D000`.
- The first tempo event is honoured; mid-song tempo changes are planned.
- PAL is the default. NTSC changes the SID clock and raster update point.
- Image conversion, PSID export, keyboard effects and packed pattern data are
  planned follow-up features.

## Tests

```sh
python3 -m unittest -v
```

## Licence

MIT. See [LICENSE](LICENSE).
