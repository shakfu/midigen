import argparse
import re
from pathlib import Path

OUT_DIR = Path("midi")

MUSICIAN_MODEL = "models/Musician-Llama-3.2-1B-Instruct.Q8_0.gguf"
MUSICIAN_SYSTEM = "You are a helpful music AI assistant specialized in MIDI token generation."

CHATMUSICIAN_MODEL = "models/ChatMusician.Q5_K_M.gguf"
CHATMUSICIAN_SYSTEM = "You are a music composition assistant. Respond with ABC notation only."


def _slugify(text: str, max_words: int = 6) -> str:
    words = re.findall(r"[A-Za-z0-9]+", text.lower())[:max_words]
    return "-".join(words) or "untitled"


def _resolve_out(prompt: str, out: str | None) -> Path:
    path = Path(out) if out else OUT_DIR / f"{_slugify(prompt)}.mid"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _parse_octuple_rows(text: str) -> list[list[int]]:
    rows = ([int(x) for x in tup.split() if x.lstrip("-").isdigit()]
            for tup in text.split(".") if tup.strip())
    return [r for r in rows if len(r) == 8]


def _extract_abc(text: str) -> str:
    fenced = re.search(r"```(?:abc)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    m = re.search(r"(X\s*:\s*\d.*)", text, re.DOTALL)
    if not m:
        raise ValueError(f"no ABC found in:\n{text}")
    return m.group(1).strip()


def run_musician(prompt: str, out: str | None, max_tokens: int, temperature: float) -> None:
    from cyllama import LLM
    from miditok import Octuple, TokenizerConfig

    out_path = _resolve_out(prompt, out)
    llm = LLM(MUSICIAN_MODEL, temperature=temperature, max_tokens=max_tokens)
    text = str(llm.chat([
        {"role": "system", "content": MUSICIAN_SYSTEM},
        {"role": "user", "content": prompt},
    ]))

    token_list = _parse_octuple_rows(text)

    cfg = TokenizerConfig(
        use_velocities=True,
        use_tempos=True,
        use_time_signatures=True,
        use_programs=True,
    )
    tokenizer = Octuple(cfg)
    score = tokenizer.decode(token_list)
    score.dump_midi(str(out_path))
    print(f"wrote {out_path} ({len(token_list)} tokens)")


def run_chatmusician(prompt: str, out: str | None, max_tokens: int, temperature: float) -> None:
    import subprocess
    import tempfile

    from cyllama import LLM

    out_path = _resolve_out(prompt, out)
    llm = LLM(CHATMUSICIAN_MODEL, temperature=temperature, max_tokens=max_tokens)
    text = str(llm.chat([
        {"role": "system", "content": CHATMUSICIAN_SYSTEM},
        {"role": "user", "content": prompt},
    ]))

    abc = _extract_abc(text)

    with tempfile.NamedTemporaryFile("w", suffix=".abc", delete=False) as f:
        f.write(abc)
        abc_path = f.name
    subprocess.run(
        ["abc2midi", abc_path, "-o", str(out_path)],
        check=True, capture_output=True, text=True,
    )
    print(f"wrote {out_path}")
    print(f"--- ABC ---\n{abc}\n-----------")


HANDLERS = {
    "musician": (run_musician, 2048, 0.9),
    "chatmusician": (run_chatmusician, 1024, 0.7),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MIDI from a prompt using a local LLM.")
    sub = parser.add_subparsers(dest="model", required=True)

    for name, (_, default_max_tokens, default_temp) in HANDLERS.items():
        sp = sub.add_parser(name, help=f"Use the {name} pipeline.")
        sp.add_argument("prompt", help="Natural-language prompt.")
        sp.add_argument("-o", "--out", help="Output MIDI path (default: midi/<slug>.mid).")
        sp.add_argument("--max-tokens", type=int, default=default_max_tokens)
        sp.add_argument("--temperature", type=float, default=default_temp)

    args = parser.parse_args()
    handler, _, _ = HANDLERS[args.model]
    handler(args.prompt, args.out, args.max_tokens, args.temperature)


if __name__ == "__main__":
    main()
