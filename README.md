# midigen

Generate MIDI files from natural-language prompts using local GGUF language models. Two pipelines are included, picking different symbolic-music representations.

## Pipelines

### `musician` — Musician-Llama (Octuple tokens)

- Model: [`Ghanibhuti/Musician-Llama-3.2-1B-Instruct`](https://huggingface.co/Ghanibhuti/Musician-Llama-3.2-1B-Instruct), Q8_0 GGUF (~1.2 GB).
- Representation: [miditok](https://github.com/Natooz/MidiTok) `Octuple` token streams.
- Decode: parse the model's `8 ints separated by spaces, groups separated by .` output, filter to well-formed rows, decode with `Octuple(TokenizerConfig(...))`, dump via [symusic](https://github.com/Yikai-Liao/symusic).
- Best for multi-track production-style output (drums, bass, leads) since Octuple natively carries a `Program` (GM instrument) component per event.

### `chatmusician` — ChatMusician (ABC notation)

- Model: [MaziyarPanahi/ChatMusician-GGUF](https://huggingface.co/MaziyarPanahi/ChatMusician-GGUF), [Q5_K_M GGUF](https://huggingface.co/MaziyarPanahi/ChatMusician-GGUF/resolve/main/ChatMusician.Q5_K_M.gguf?download=true) (~4.5 GB).
- Representation: ABC notation as plain text.
- Decode: extract the ABC block from the model's response and pipe through `abc2midi`.
- Best for melodic, contrapuntal, or traditional forms (reels, fugues, lead sheets) where ABC was designed to excel.

## Picking a pipeline

| If you want                                                       | Use             |
|-------------------------------------------------------------------|-----------------|
| Multi-track grooves (EDM, hip-hop, rock); per-note velocity       | Musician-Llama  |
| Melody, counterpoint, traditional forms; lowest decode friction   | ChatMusician    |
| Smallest footprint / fastest iteration                            | Musician-Llama  |
| Natural-language prompts with rich structural control             | ChatMusician    |

## Setup

```sh
uv sync
brew install abcmidi   # for the ChatMusician pipeline
```

Place model files in `models/`:

```
models/Musician-Llama-3.2-1B-Instruct.Q8_0.gguf
models/ChatMusician.Q5_K_M.gguf
```

## Usage

A single `midigen.py` CLI dispatches to either pipeline. Each pipeline's heavy imports are loaded lazily, so picking one does not pay the cost of the other.

```sh
uv run midigen.py musician     "Upbeat electronic dance music with strong bass and drum patterns"
uv run midigen.py chatmusician "Compose a 16-bar Irish reel in G major"
```

Both write to `midi/<slugified-prompt>.mid` by default. Override with `-o path.mid`. Tune generation with `--temperature` and `--max-tokens`. Run `uv run midigen.py --help` (or `... musician --help`) for full options.

## Dependencies

- [`cyllama`](https://github.com/shakfu/cyllama) — Pythonic llama.cpp wrapper, drives GGUF inference for both pipelines.
- [`miditok`](https://github.com/Natooz/MidiTok) — symbolic-music tokenization; used to decode Musician-Llama's Octuple output.
- `abcmidi` (Homebrew) — provides `abc2midi`, used to render ChatMusician's ABC output.

## Notes

The Musician-Llama tokenizer configuration is inferred (the HF repo does not ship a miditok `tokenizer.json`), so musical quality depends on whether the inferred bin sizes match those used at training time. Auditory verification on a dance-music prompt produced coherent drums-plus-synth-bass output, suggesting the inference is close.

The model's documented output format on the HF model card (pipe-separated 4-tuples) is incorrect; the actual format is space-separated 8-ints per group, separated by `.`.
