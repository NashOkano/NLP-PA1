"""Required decoding entry point: run_tagger.py SENT_FILE MODEL_FILE OUTPUT_FILE."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from pos_tagger.hmm import HMMTagger


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("Usage: python run_tagger.py SENT_FILE MODEL_FILE OUTPUT_FILE")
    sentences = Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
    tagger = HMMTagger.load(Path(sys.argv[2]))
    # The assignment's reference file and scorer expect ``word/TAG`` tokens,
    # not a tag-only sequence.
    output = [
        " ".join(f"{word}/{tag}" for word, tag in zip(sentence.split(), tagger.tag(sentence.split())))
        for sentence in sentences
    ]
    Path(sys.argv[3]).write_text("\n".join(output) + ("\n" if output else ""), encoding="utf-8")


if __name__ == "__main__":
    main()
