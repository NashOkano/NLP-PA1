"""Required training entry point: build_tagger.py TRAIN_FILE MODEL_FILE."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from pos_tagger.hmm import HMMTagger


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python build_tagger.py TRAIN_FILE MODEL_FILE")
    tagger = HMMTagger.train_from_file(Path(sys.argv[1]))
    tagger.save(Path(sys.argv[2]))


if __name__ == "__main__":
    main()
