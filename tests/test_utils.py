"""Tests for yololabeler.utils, EXIF-driven image auto-orientation."""

import io

from PIL import Image

from yololabeler.utils import auto_orient_image, is_image_file, oriented_size

RED = (255, 0, 0)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)


def make_image(orientation=None, size=(4, 2), extra_exif=None):
    """Build a 4x2 PNG (red top-left, green top-right) with optional EXIF."""
    img = Image.new("RGB", size, WHITE)
    img.putpixel((0, 0), RED)
    img.putpixel((size[0] - 1, 0), GREEN)
    buf = io.BytesIO()
    if orientation is None and extra_exif is None:
        img.save(buf, format="PNG")
    else:
        exif = Image.Exif()
        if orientation is not None:
            exif[274] = orientation
        if extra_exif:
            exif.update(extra_exif)
        img.save(buf, format="PNG", exif=exif)
    buf.seek(0)
    return Image.open(buf)


def corner(img, x, y):
    return img.convert("RGB").load()[x, y]


# ── auto_orient_image: images without usable EXIF ───────────────────────────


class TestAutoOrientNoExif:
    def test_plain_image_returned_unchanged(self):
        """A bare Image has no _getexif; the error is swallowed."""
        img = Image.new("RGB", (10, 20))
        result = auto_orient_image(img)
        assert result is img
        assert result.size == (10, 20)

    def test_file_without_exif_block(self):
        img = make_image(orientation=None)
        result = auto_orient_image(img)
        assert result is img
        assert result.size == (4, 2)

    def test_exif_without_orientation_tag(self):
        img = make_image(orientation=None, extra_exif={271: "TestCamera"})
        result = auto_orient_image(img)
        assert result is img
        assert result.size == (4, 2)

    def test_unknown_orientation_value_ignored(self):
        img = make_image(orientation=99)
        result = auto_orient_image(img)
        assert result.size == (4, 2)
        assert corner(result, 0, 0) == RED
        assert corner(result, 3, 0) == GREEN


# ── auto_orient_image: orientations that keep the aspect ratio ──────────────


class TestAutoOrientUpright:
    def test_orientation_1_is_a_no_op(self):
        result = auto_orient_image(make_image(1))
        assert result.size == (4, 2)
        assert corner(result, 0, 0) == RED
        assert corner(result, 3, 0) == GREEN

    def test_orientation_2_mirrors_horizontally(self):
        result = auto_orient_image(make_image(2))
        assert result.size == (4, 2)
        assert corner(result, 0, 0) == GREEN
        assert corner(result, 3, 0) == RED

    def test_orientation_3_rotates_180(self):
        result = auto_orient_image(make_image(3))
        assert result.size == (4, 2)
        assert corner(result, 3, 1) == RED
        assert corner(result, 0, 1) == GREEN

    def test_orientation_4_mirrors_vertically(self):
        result = auto_orient_image(make_image(4))
        assert result.size == (4, 2)
        assert corner(result, 0, 1) == RED
        assert corner(result, 3, 1) == GREEN


# ── auto_orient_image: orientations that swap width and height ──────────────


class TestAutoOrientTransposed:
    def test_orientation_5_transposes(self):
        """Mirror horizontal + rotate 270 CW is a main-diagonal flip (x, y) ->
        (y, x).
        """
        result = auto_orient_image(make_image(5))
        assert result.size == (2, 4)
        assert corner(result, 0, 0) == RED
        assert corner(result, 0, 3) == GREEN

    def test_orientation_6_rotates_90_clockwise(self):
        result = auto_orient_image(make_image(6))
        assert result.size == (2, 4)
        assert corner(result, 1, 0) == RED
        assert corner(result, 1, 3) == GREEN

    def test_orientation_7_transverses(self):
        """Mirror horizontal + rotate 90 CW is an anti-diagonal flip."""
        result = auto_orient_image(make_image(7))
        assert result.size == (2, 4)
        assert corner(result, 1, 3) == RED
        assert corner(result, 1, 0) == GREEN

    def test_orientation_8_rotates_90_counterclockwise(self):
        result = auto_orient_image(make_image(8))
        assert result.size == (2, 4)
        assert corner(result, 0, 3) == RED
        assert corner(result, 0, 0) == GREEN

    def test_transposed_result_is_a_new_image(self):
        img = make_image(6)
        result = auto_orient_image(img)
        assert result is not img


# ── oriented_size ───────────────────────────────────────────────────────────


class TestOrientedSize:
    def test_no_exif(self, tmp_path):
        p = tmp_path / "a.jpg"
        Image.new("RGB", (40, 20)).save(p)
        assert oriented_size(p) == (40, 20, 1)

    def test_rotated_swaps_dimensions(self, tmp_path):
        p = tmp_path / "a.jpg"
        exif = Image.Exif()
        exif[274] = 6
        Image.new("RGB", (40, 20)).save(p, exif=exif)
        assert oriented_size(p) == (20, 40, 6)


# ── is_image_file ────────────────────────────────────────────────────────────


class TestIsImageFile:
    def test_accepts_known_extensions(self):
        for name in ("a.png", "b.JPG", "c.jpeg", "d.bmp", "e.tif", "f.TIFF"):
            assert is_image_file(name)

    def test_rejects_other_extensions(self):
        assert not is_image_file("readme.txt")
        assert not is_image_file("notes.json")

    def test_rejects_macos_appledouble_files(self):
        assert not is_image_file("._IMG_0001.jpg")

    def test_rejects_dotfiles(self):
        assert not is_image_file(".DS_Store")
        assert not is_image_file(".hidden.png")
