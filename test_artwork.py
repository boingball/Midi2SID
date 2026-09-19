"""Tests for JPG/PNG background artwork conversion.

Pillow is an optional dependency (only --artwork needs it), so these tests
skip cleanly if it isn't installed rather than failing the whole suite.
"""
import tempfile
import unittest
from pathlib import Path

import build_prg
import sid_midi
from cpu6502_sim import CPU

try:
    from PIL import Image
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False

if HAVE_PIL:
    import artwork


def _make_test_image(path: Path, size=(60, 40)) -> None:
    image = Image.new("RGB", size)
    pixels = image.load()
    w, h = size
    for y in range(h):
        for x in range(w):
            pixels[x, y] = (255 * x // w, 255 * y // h, 128)
    image.save(path)


@unittest.skipUnless(HAVE_PIL, "Pillow not installed; --artwork is optional")
class ArtworkConversionTests(unittest.TestCase):
    def test_render_artwork_produces_correctly_sized_buffers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.png"
            _make_test_image(path)
            bitmap, screen = artwork.render_artwork(path)
        self.assertEqual(len(bitmap), artwork.BITMAP_SIZE)
        self.assertEqual(len(screen), 1024)

    def test_render_artwork_uses_only_valid_palette_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.png"
            _make_test_image(path)
            _, screen = artwork.render_artwork(path)
        for nibble in screen[:artwork.SCREEN_SIZE]:
            self.assertLess(nibble >> 4, 16)
            self.assertLess(nibble & 0xf, 16)

    def test_odd_aspect_ratio_image_does_not_crash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tall.png"
            _make_test_image(path, size=(20, 200))
            bitmap, screen = artwork.render_artwork(path)
        self.assertEqual(len(bitmap), artwork.BITMAP_SIZE)


@unittest.skipUnless(HAVE_PIL, "Pillow not installed; --artwork is optional")
class ArtworkPlayerIntegrationTests(unittest.TestCase):
    def _build(self, artwork_path):
        frame = bytes([0] * 24 + [15])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "art.prg"
            art = artwork.render_artwork(artwork_path) if artwork_path else None
            build_prg.build_prg([frame], output, "ART", artwork=art)
            return output.read_bytes(), art

    def test_artwork_bitmap_bytes_are_embedded_in_the_prg(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.png"
            _make_test_image(path)
            data, art = self._build(path)
        self.assertIn(art[0], data)

    def test_boot_copies_artwork_screen_colours_instead_of_uniform_fill(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.png"
            _make_test_image(path)
            art = artwork.render_artwork(path)

        captured = {}
        original_resolve = build_prg.Assembler.resolve

        def resolve(self):
            captured["labels"] = dict(self.labels)
            return original_resolve(self)

        build_prg.Assembler.resolve = resolve
        try:
            reg = bytearray(25)
            frames = [sid_midi.SidFrame(reg, 0)]
            packed = build_prg.pack_lzss(build_prg.encode_events(frames))
            code = build_prg._player("ART", packed, "pal", artwork=art)
        finally:
            build_prg.Assembler.resolve = original_resolve

        mem = bytearray(65536)
        mem[build_prg.ENTRY_ADDRESS:build_prg.ENTRY_ADDRESS + len(code)] = code
        rom = bytearray(4096)
        for screencode in range(256):
            for row in range(8):
                rom[screencode * 8 + row] = screencode
        cpu = CPU(mem, char_rom=bytes(rom))
        try:
            cpu.run(build_prg.ENTRY_ADDRESS, max_steps=500_000)
        except RuntimeError:
            pass

        artwork_screen = art[1]
        # A cell far from any label/scope row should keep the artwork's own
        # colour, not the uniform INK_PAPER fallback fill.
        untouched_row, untouched_col = 18, 30
        offset = untouched_row * 40 + untouched_col
        self.assertEqual(mem[0x0400 + offset], artwork_screen[offset])

        # But a label cell must still be forced to the fixed, legible colour
        # regardless of what the artwork put there, so text stays readable.
        logo_row, logo_col = 0, 16
        self.assertEqual(mem[0x0400 + logo_row * 40 + logo_col], build_prg.INK_PAPER)

        # And the scope row's colour is likewise forced, not the artwork's.
        scope_row, scope_col = 6, 2
        self.assertEqual(mem[0x0400 + scope_row * 40 + scope_col], build_prg.INK_PAPER)


if __name__ == "__main__":
    unittest.main()
