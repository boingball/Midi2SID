# MIDI2SID

Smart MIDI-to-Commodore 64 SID conversion, inspired by the voice-routing work
in [MIDI2AY](https://github.com/boingball/Midi2AY).

MIDI2SID reads format 0/1 PPQN Standard MIDI files and creates a self-running
C64 `.prg`. It automatically reduces polyphonic arrangements to the SID's three
voices, maps all 128 General MIDI programs to SID synth families, and gives MIDI
channel 10 a dedicated synthesized drum path with distinct tuned or noise
treatment for kicks, snares, toms, hats, cymbals, cowbell/claves/woodblocks/
triangle, bongos/congas/timbales/cuica, agogo/ride bell, and guiro/vibraslap;
anything else on channel 10 still falls back to generic noise.

Before rendering individual frames, the arranger scores each non-drum MIDI
channel across the complete song to identify its likely melody. SID voice one
then follows that channel whenever it is active, voice two favours sustained
harmonic backing, and voice three favours a genuine General MIDI bass part.
Role-balanced ADSR sustain leaves headroom for the melody instead of allowing
bass and busy arpeggios to mask it.

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
python3 midi2sid.py song.mid song.prg --feel expressive
python3 midi2sid.py song.mid song.prg --artwork cover.jpg
python3 midi2sid.py song.mid song.prg --channel-report
python3 midi2sid.py song.mid song.prg --exclude-channels 10
python3 midi2sid.py song.mid song.prg --trim-seconds 90
```

`--artwork` needs [Pillow](https://python-pillow.org/) (`pip install Pillow`) to
decode and resize the image; it is the only optional dependency and only
needed if you use that flag.

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

The default `--feel tight` mode gives every selected MIDI note a real SID gate
retrigger, writes ADSR before raising the gate, uses immediate attacks and keeps
drum voice-stealing short. It also holds each patch's pulse width steady instead
of restarting PWM on every short note. This is intended for rhythmically precise
automatic conversions. `--feel expressive` restores animated PWM and the slower
family-specific attack/release values for files arranged around sustained pads
and strings.

The generated player starts in visual mode 2. The whole screen is a real
VIC-II hi-res bitmap (not text mode): the title and labels are blitted from
the real character ROM at boot, and each SID voice gets a genuine travelling
pixel oscilloscope trace shaped to match what that voice is actually
synthesising: a smooth ramp for triangle, a rising ramp for sawtooth, a
square-edged trace for pulse, or a jagged static-like trace for noise, read
live from the voice's own SID control register. This is the same
phase-cycling-picture idea MIDI2AY uses for its pitch-reactive Spectrum
scope, just picking one of four shapes instead of one shared wiggle.
During playback, keys 1-5 select: safe/static, scopes, slow border, scopes
plus border, and slow demo mode (modes 4 and 5 currently look the same,
since the old background-colour pulse in mode 5 has no visible effect once
the screen is bitmap). The scope animation follows each oscillator's gate,
frequency and waveform while colour changes remain deliberately slow.

This is an automatic chip-music arrangement, not a transparent reproduction of
the source MIDI. Dense chords must be reduced to three voices, and drums borrow
voice three while they sound.

## Background artwork

`--artwork picture.jpg` (or `.png`) dithers an image to fill the whole
320x200 bitmap behind the title, labels and scope, the same "full-screen
picture with UI overlaid" idea as MIDI2AY's title card. VIC-II hi-res mode
allows exactly two colours per 8x8 cell (the same constraint as the
Spectrum's attribute clash), so each cell picks its own best two colours
from the C64's 16-colour palette, then the whole image is Floyd-Steinberg
dithered against those fixed per-cell palettes - the classic technique
behind hand-digitised C64 "hires" photos. The image is cropped to fill the
320x200 frame (not squashed). Label text and the oscilloscope rows always
force their own cells back to a fixed, legible ink/paper colour after the
artwork is drawn, so they stay readable regardless of what the picture put
there.

## Songs too large for PRG v1

The PRG v1 player combines register deltas with a streaming, 256-byte-window
LZSS decoder, which only catches repeats within that 256-byte lookback, not
whole repeated bars or choruses further apart. Very long or busy songs (dense
drum tracks especially) can still exceed the format's `$D000` budget and fail
to build with a "song is too large" error.

Three flags help right now:

- `--channel-report` prints each MIDI channel's note count, how much of the
  song it's active for, and its instrument family, without building anything.
  Use it to see which channels are actually driving the size.
- `--exclude-channels 3,10` drops the listed MIDI channels (1-16, so channel
  10 is the standard GM drum channel) before conversion.
- `--trim-seconds 90` keeps only the first N seconds of the song.

Both can be combined, and used with `--drums off` too. A pattern-bank backend
(reusing repeated bars, like an Amiga MOD's pattern table) is the planned fix
that would lift the ceiling itself instead of asking you to cut material, but
it needs a new packed-event format and player-side dispatcher, so it's a
larger follow-up project rather than a quick patch.

## Current limits
- The oscilloscope bitmap reserves a fixed 8000-byte VIC-II bitmap at
  `$2000-$3F3F` (forced by hardware alignment), plus ~6.1KB for the four
  waveform-shape (triangle/saw/pulse/noise) picture sets right after it, so
  the packed song event budget is smaller than before: roughly 30 KB instead
  of the old ~50 KB single contiguous region. `--artwork` costs another ~1KB
  (the per-cell colour table) on top of that.
- The first tempo event is honoured; mid-song tempo changes are planned.
- PAL is the default. NTSC changes the SID clock and raster update point.
- PSID export, keyboard effects and packed pattern data are planned
  follow-up features.

## Tests

```sh
python3 -m unittest -v
```

GitHub Actions runs the same test suite for every push and pull request.

## Licence

MIT. See [LICENSE](LICENSE).
