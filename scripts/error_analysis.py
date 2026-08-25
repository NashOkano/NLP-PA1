"""Report POS-tagging errors by known/OOV status and tag confusion."""
from collections import Counter
from pathlib import Path
import sys


def split_word_tag(token: str) -> tuple[str, str]:
    word, tag = token.rsplit("/", 1)
    return word, tag


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("Usage: python scripts/error_analysis.py TRAIN PREDICTIONS GOLD")

    train_vocabulary = {
        split_word_tag(token)[0]
        for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
        for token in line.split()
    }
    predictions = Path(sys.argv[2]).read_text(encoding="utf-8").splitlines()
    gold = Path(sys.argv[3]).read_text(encoding="utf-8").splitlines()
    if len(predictions) != len(gold):
        raise SystemExit("Prediction and gold files have different numbers of sentences.")

    totals = Counter()
    correct = Counter()
    confusions: Counter[tuple[str, str]] = Counter()
    oov_errors: Counter[tuple[str, str, str]] = Counter()
    for line_number, (prediction_line, gold_line) in enumerate(zip(predictions, gold), 1):
        predicted = [split_word_tag(token) for token in prediction_line.split()]
        actual = [split_word_tag(token) for token in gold_line.split()]
        if len(predicted) != len(actual):
            raise SystemExit(f"Sentence {line_number} has a different number of predicted and gold tokens.")
        for (predicted_word, predicted_tag), (gold_word, gold_tag) in zip(predicted, actual):
            if predicted_word != gold_word:
                raise SystemExit(f"Word mismatch on sentence {line_number}: {predicted_word!r} vs {gold_word!r}")
            group = "known" if gold_word in train_vocabulary else "oov"
            totals[group] += 1
            if predicted_tag == gold_tag:
                correct[group] += 1
            else:
                confusions[(gold_tag, predicted_tag)] += 1
                if group == "oov":
                    oov_errors[(gold_word, gold_tag, predicted_tag)] += 1

    for group in ("known", "oov"):
        print(f"{group.upper():5}: {correct[group]}/{totals[group]} = {correct[group] / totals[group]:.2%}")
    print("\nMost frequent tag confusions (gold -> predicted):")
    for (gold_tag, predicted_tag), count in confusions.most_common(15):
        print(f"  {gold_tag:>4} -> {predicted_tag:<4} {count}")
    print("\nMost frequent OOV errors (word: gold -> predicted):")
    for (word, gold_tag, predicted_tag), count in oov_errors.most_common(15):
        print(f"  {word}: {gold_tag} -> {predicted_tag} ({count})")


if __name__ == "__main__":
    main()
