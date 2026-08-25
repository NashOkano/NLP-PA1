"""Evaluate a tag-only output file against the assignment's dev.tags reference."""
from pathlib import Path
import sys


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python scripts/evaluate.py PREDICTIONS GOLD")
    prediction_lines = Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
    gold_lines = Path(sys.argv[2]).read_text(encoding="utf-8").splitlines()
    if len(prediction_lines) != len(gold_lines):
        raise SystemExit("Prediction and gold files have different numbers of sentences.")
    correct = total = 0
    for number, (prediction, gold) in enumerate(zip(prediction_lines, gold_lines), 1):
        predicted_tags = [item.rsplit("/", 1)[-1] for item in prediction.split()]
        gold_tags = [item.rsplit("/", 1)[-1] for item in gold.split()]
        if len(predicted_tags) != len(gold_tags):
            raise SystemExit(f"Sentence {number} has {len(predicted_tags)} predicted tags; expected {len(gold_tags)}.")
        correct += sum(a == b for a, b in zip(predicted_tags, gold_tags))
        total += len(gold_tags)
    print(f"Accuracy = {correct / total:.4%} ({correct}/{total})")


if __name__ == "__main__":
    main()
