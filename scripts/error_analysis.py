from collections import Counter
from pathlib import Path
import sys


def split_word_tag(token: str) -> tuple[str, str]:
    word, tag = token.rsplit("/", 1)
    return word, tag


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(
            "Usage: python scripts/error_analysis.py TRAIN PREDICTIONS GOLD"
        )

    train_vocabulary = {
        split_word_tag(token)[0]
        for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
        for token in line.split()
    }

    predictions = Path(sys.argv[2]).read_text(encoding="utf-8").splitlines()
    gold = Path(sys.argv[3]).read_text(encoding="utf-8").splitlines()

    if len(predictions) != len(gold):
        raise SystemExit(
            "Prediction and gold files have different numbers of sentences."
        )

    totals = Counter()
    correct = Counter()

    # Overall confusion counts
    confusions: Counter[tuple[str, str]] = Counter()

    # Confusion counts split into known/OOV
    group_confusions: Counter[tuple[str, str, str]] = Counter()

    # Individual error words
    known_errors: Counter[tuple[str, str, str]] = Counter()
    oov_errors: Counter[tuple[str, str, str]] = Counter()

    for line_number, (prediction_line, gold_line) in enumerate(
        zip(predictions, gold), 1
    ):
        predicted = [
            split_word_tag(token) for token in prediction_line.split()
        ]
        actual = [
            split_word_tag(token) for token in gold_line.split()
        ]

        if len(predicted) != len(actual):
            raise SystemExit(
                f"Sentence {line_number} has a different number of "
                "predicted and gold tokens."
            )

        for (
            (predicted_word, predicted_tag),
            (gold_word, gold_tag),
        ) in zip(predicted, actual):

            if predicted_word != gold_word:
                raise SystemExit(
                    f"Word mismatch on sentence {line_number}: "
                    f"{predicted_word!r} vs {gold_word!r}"
                )

            group = (
                "known"
                if gold_word in train_vocabulary
                else "oov"
            )

            totals[group] += 1

            if predicted_tag == gold_tag:
                correct[group] += 1

            else:
                # Overall confusion
                confusions[(gold_tag, predicted_tag)] += 1

                # Known/OOV confusion breakdown
                group_confusions[
                    (gold_tag, predicted_tag, group)
                ] += 1

                # Individual error words
                if group == "known":
                    known_errors[
                        (gold_word, gold_tag, predicted_tag)
                    ] += 1
                else:
                    oov_errors[
                        (gold_word, gold_tag, predicted_tag)
                    ] += 1

    # ------------------------------------------------------------------
    # Accuracy
    # ------------------------------------------------------------------

    total_tokens = totals["known"] + totals["oov"]
    total_correct = correct["known"] + correct["oov"]

    print(
        f"Accuracy = {total_correct / total_tokens:.4%} "
        f"({total_correct}/{total_tokens})"
    )

    for group in ("known", "oov"):
        print(
            f"{group.upper():5}: "
            f"{correct[group]}/{totals[group]} = "
            f"{correct[group] / totals[group]:.2%}"
        )

    # ------------------------------------------------------------------
    # Overall tag confusions
    # ------------------------------------------------------------------

    print("\nMost frequent tag confusions (gold -> predicted):")

    for (gold_tag, predicted_tag), count in confusions.most_common(15):
        print(
            f"  {gold_tag:>4} -> {predicted_tag:<4} {count}"
        )

    # ------------------------------------------------------------------
    # Known/OOV breakdown of major confusions
    # ------------------------------------------------------------------

    print("\nTop confusions split by known/OOV:")

    for (gold_tag, predicted_tag), count in confusions.most_common(15):
        known = group_confusions[
            (gold_tag, predicted_tag, "known")
        ]
        oov = group_confusions[
            (gold_tag, predicted_tag, "oov")
        ]

        print(
            f"  {gold_tag:>4} -> {predicted_tag:<4} "
            f"total={count:<4} "
            f"known={known:<4} "
            f"oov={oov:<4}"
        )

    # ------------------------------------------------------------------
    # Most frequent OOV errors
    # ------------------------------------------------------------------

    print("\nMost frequent OOV errors (word: gold -> predicted):")

    for (word, gold_tag, predicted_tag), count in oov_errors.most_common(15):
        print(
            f"  {word}: {gold_tag} -> {predicted_tag} ({count})"
        )

    # ------------------------------------------------------------------
    # Most frequent known NN -> JJ errors
    # ------------------------------------------------------------------

    print("\nMost frequent known NN -> JJ errors:")

    found_known_nn_jj = False

    for (word, gold_tag, predicted_tag), count in known_errors.most_common():
        if gold_tag == "NN" and predicted_tag == "JJ":
            print(
                f"  {word}: {gold_tag} -> {predicted_tag} ({count})"
            )
            found_known_nn_jj = True

    if not found_known_nn_jj:
        print("  None")


if __name__ == "__main__":
    main()