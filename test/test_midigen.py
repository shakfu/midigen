import shutil
from pathlib import Path

import pytest

from midigen import (
    CHATMUSICIAN_MODEL,
    MUSICIAN_MODEL,
    OUT_DIR,
    _extract_abc,
    _parse_octuple_rows,
    _resolve_out,
    _slugify,
    run_chatmusician,
    run_musician,
)


# --- pure-function unit tests --------------------------------------------------


class TestSlugify:
    def test_basic(self):
        assert _slugify("Hello World") == "hello-world"

    def test_strips_punctuation(self):
        assert _slugify("Upbeat! EDM, with bass.") == "upbeat-edm-with-bass"

    def test_truncates_to_max_words(self):
        text = "one two three four five six seven eight"
        assert _slugify(text, max_words=3) == "one-two-three"

    def test_default_max_words_is_six(self):
        text = "one two three four five six seven"
        assert _slugify(text) == "one-two-three-four-five-six"

    def test_empty_returns_untitled(self):
        assert _slugify("") == "untitled"

    def test_punctuation_only_returns_untitled(self):
        assert _slugify("!!! ... ???") == "untitled"

    def test_lowercases(self):
        assert _slugify("ALLCAPS Mixed") == "allcaps-mixed"

    def test_keeps_digits(self):
        assert _slugify("16 bar reel in G3") == "16-bar-reel-in-g3"


class TestResolveOut:
    def test_uses_explicit_out_path(self, tmp_path):
        target = tmp_path / "custom" / "name.mid"
        result = _resolve_out("ignored prompt", str(target))
        assert result == target
        assert result.parent.exists()

    def test_defaults_from_prompt_slug(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _resolve_out("Hello World Test", None)
        assert result == Path("midi") / "hello-world-test.mid"
        assert result.parent.exists()

    def test_creates_nested_parent(self, tmp_path):
        target = tmp_path / "a" / "b" / "c" / "out.mid"
        _resolve_out("p", str(target))
        assert target.parent.is_dir()


class TestParseOctupleRows:
    def test_well_formed_eight_int_rows(self):
        text = "40 4 4 28 4 37 16 12. 40 6 4 28 4 37 16 12"
        result = _parse_octuple_rows(text)
        assert result == [
            [40, 4, 4, 28, 4, 37, 16, 12],
            [40, 6, 4, 28, 4, 37, 16, 12],
        ]

    def test_trailing_dot_is_tolerated(self):
        text = "40 4 4 28 4 37 16 12."
        assert _parse_octuple_rows(text) == [[40, 4, 4, 28, 4, 37, 16, 12]]

    def test_rows_with_wrong_arity_are_dropped(self):
        text = "40 4 4 28 4 37 16 12. 1 2 3. 40 6 4 28 4 37 16 12"
        result = _parse_octuple_rows(text)
        assert len(result) == 2
        assert all(len(r) == 8 for r in result)

    def test_empty_input_returns_empty_list(self):
        assert _parse_octuple_rows("") == []

    def test_whitespace_only_input(self):
        assert _parse_octuple_rows("   .  .  ") == []

    def test_handles_negative_ints(self):
        text = "40 -4 4 28 4 37 16 12"
        assert _parse_octuple_rows(text) == [[40, -4, 4, 28, 4, 37, 16, 12]]

    def test_skips_non_digit_garbage_in_a_row(self):
        text = "40 4 foo 28 4 37 16 12"
        assert _parse_octuple_rows(text) == []


class TestExtractAbc:
    def test_fenced_abc_block(self):
        text = "Here is your tune:\n```abc\nX:1\nK:G\nGABc|\n```\nEnjoy!"
        assert _extract_abc(text) == "X:1\nK:G\nGABc|"

    def test_fenced_block_without_language_tag(self):
        text = "```\nX:1\nK:C\nCDEF|\n```"
        assert _extract_abc(text) == "X:1\nK:C\nCDEF|"

    def test_bare_abc_starting_with_x_header(self):
        text = "Sure! X:1\nL:1/8\nM:4/4\nK:G\nGABc|"
        result = _extract_abc(text)
        assert result.startswith("X:1")
        assert "GABc|" in result

    def test_x_header_with_whitespace_variants(self):
        text = "X : 2\nK:D\nDEFG|"
        result = _extract_abc(text)
        assert result.startswith("X : 2")

    def test_raises_when_no_abc_present(self):
        with pytest.raises(ValueError, match="no ABC found"):
            _extract_abc("Just some prose without any music notation.")

    def test_prefers_fenced_block_over_bare_header(self):
        text = "X:99\nK:Bogus\n```abc\nX:1\nK:G\nGABc|\n```"
        result = _extract_abc(text)
        assert "X:1" in result
        assert "X:99" not in result


# --- real-model integration tests ---------------------------------------------
#
# These load the actual GGUF models and run inference, writing real MIDI files
# to the project's midi/ directory. They are slow (seconds for Musician-Llama,
# tens of seconds for ChatMusician) and nondeterministic (temperature > 0).
# Skip with: uv run pytest -m "not slow"


def _midi_has_valid_header(path: Path) -> bool:
    """A MIDI file begins with the bytes 'MThd'."""
    with open(path, "rb") as f:
        return f.read(4) == b"MThd"


@pytest.fixture(scope="session")
def musician_model_available():
    if not Path(MUSICIAN_MODEL).exists():
        pytest.skip(f"{MUSICIAN_MODEL} not present")


@pytest.fixture(scope="session")
def chatmusician_model_available():
    if not Path(CHATMUSICIAN_MODEL).exists():
        pytest.skip(f"{CHATMUSICIAN_MODEL} not present")


@pytest.fixture(scope="session")
def abc2midi_available():
    if shutil.which("abc2midi") is None:
        pytest.skip("abc2midi not installed (brew install abcmidi)")


@pytest.mark.slow
class TestRunMusicianReal:
    def test_default_path_writes_valid_midi(self, musician_model_available):
        prompt = "test musician default upbeat bass drum"
        out = OUT_DIR / f"{_slugify(prompt)}.mid"
        out.unlink(missing_ok=True)

        run_musician(prompt, out=None, max_tokens=512, temperature=0.9)

        assert out.is_file(), f"expected output at {out}"
        assert out.stat().st_size > 0
        assert _midi_has_valid_header(out)

    def test_explicit_out_writes_valid_midi(self, musician_model_available):
        out = OUT_DIR / "test_musician_explicit.mid"
        out.unlink(missing_ok=True)

        run_musician(
            "ambient pad with slow evolving texture",
            out=str(out), max_tokens=512, temperature=0.9,
        )

        assert out.is_file()
        assert _midi_has_valid_header(out)


@pytest.mark.slow
class TestRunChatmusicianReal:
    def test_default_path_writes_valid_midi(
        self, chatmusician_model_available, abc2midi_available
    ):
        prompt = "test chatmusician compose short tune G major"
        out = OUT_DIR / f"{_slugify(prompt)}.mid"
        out.unlink(missing_ok=True)

        run_chatmusician(prompt, out=None, max_tokens=512, temperature=0.7)

        assert out.is_file(), f"expected output at {out}"
        assert out.stat().st_size > 0
        assert _midi_has_valid_header(out)

    def test_explicit_out_writes_valid_midi(
        self, chatmusician_model_available, abc2midi_available
    ):
        out = OUT_DIR / "test_chatmusician_explicit.mid"
        out.unlink(missing_ok=True)

        run_chatmusician(
            "Compose a short 8-bar melody in D minor.",
            out=str(out), max_tokens=512, temperature=0.7,
        )

        assert out.is_file()
        assert _midi_has_valid_header(out)
