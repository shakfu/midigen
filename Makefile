.PHONY: help install test test-all musician chatmusician clean

PROMPT ?= Upbeat electronic dance music with strong bass and drum patterns

help:
	@echo "Targets:"
	@echo "  install       Sync deps (including dev group)."
	@echo "  test          Run fast tests only (skip model-loading integration tests)."
	@echo "  test-all      Run every test, including slow model-loading ones."
	@echo "  musician      Run the Musician-Llama pipeline. Override prompt with PROMPT=..."
	@echo "  chatmusician  Run the ChatMusician pipeline. Override prompt with PROMPT=..."
	@echo "  clean         Remove generated MIDI, caches, and __pycache__ dirs."

install:
	uv sync --all-groups

test:
	uv run pytest -m "not slow"

test-all:
	uv run pytest

musician:
	uv run midigen.py musician "$(PROMPT)"

chatmusician:
	uv run midigen.py chatmusician "$(PROMPT)"

clean:
	rm -rf midi/*.mid .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
