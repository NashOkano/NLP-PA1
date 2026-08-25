"""Smoothed bigram HMM with shape-aware, backoff suffix OOV handling."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Iterable

START = "<START>"
END = "<END>"
TRANSITION_ALPHA = 0.1
EMISSION_ALPHA = 0.01
SUFFIX_ALPHA = 0.1
MAX_SUFFIX_LENGTH = 5
RARE_WORD_LIMIT = 4
MIN_SUFFIX_OBSERVATIONS = 2
SUFFIX_BACKOFF_STRENGTH = 20.0
OOV_TAG_CONTEXT_WEIGHT = 0.35
OOV_CONTEXT_MARGIN = 0.05


def word_shape(word: str, position: int) -> str:
    """Return an OOV class; initial capitalization is deliberately ignored."""
    if any(character.isdigit() for character in word):
        return "HAS_DIGIT"
    if "-" in word:
        return "HYPHENATED"
    if len(word) > 1 and word.isupper():
        return "ALL_CAPS"
    if position > 0 and word[:1].isupper():
        return "CAPITALIZED"
    return "LOWER"


def _read_tagged_sentences(path: Path) -> Iterable[list[tuple[str, str]]]:
    """Read the assignment's whitespace-separated word/tag sentence format."""
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        sentence: list[tuple[str, str]] = []
        for item in line.split():
            if "/" not in item:
                raise ValueError(f"Malformed token on line {line_number}: {item!r}")
            word, tag = item.rsplit("/", 1)
            if not word or not tag:
                raise ValueError(f"Malformed word/tag token on line {line_number}: {item!r}")
            sentence.append((word, tag))
        if sentence:
            yield sentence


class HMMTagger:
    def __init__(self, model: dict):
        self.model = model
        self.tags = model["tags"]
        self.tag_set = set(self.tags)
        self.vocabulary = set(model["word_tags"])

    @classmethod
    def train_from_file(cls, path: Path) -> "HMMTagger":
        return cls.train(_read_tagged_sentences(path))

    @classmethod
    def train(cls, sentences: Iterable[list[tuple[str, str]]]) -> "HMMTagger":
        materialized = list(sentences)
        word_frequency = Counter(word for sent in materialized for word, _ in sent)
        tag_counts: Counter[str] = Counter()
        transition_counts: Counter[tuple[str, str]] = Counter()
        outgoing_counts: Counter[str] = Counter()
        emission_counts: Counter[tuple[str, str]] = Counter()
        word_tags: dict[str, set[str]] = defaultdict(set)
        suffix_counts: Counter[tuple[str, str, str]] = Counter()
        suffix_totals: Counter[tuple[str, str]] = Counter()
        shape_tag_counts: Counter[tuple[str, str]] = Counter()
        shape_totals: Counter[str] = Counter()
        tag_context_counts: Counter[tuple[str, str, str]] = Counter()
        tag_context_totals: Counter[tuple[str, str]] = Counter()

        for sentence in materialized:
            previous = START
            for position, (word, tag) in enumerate(sentence):
                tag_counts[tag] += 1
                transition_counts[(previous, tag)] += 1
                outgoing_counts[previous] += 1
                emission_counts[(tag, word)] += 1
                word_tags[word].add(tag)
                previous_tag = sentence[position - 1][1] if position else START
                next_tag = sentence[position + 1][1] if position + 1 < len(sentence) else END
                tag_context_counts[(previous_tag, next_tag, tag)] += 1
                tag_context_totals[(previous_tag, next_tag)] += 1
                previous = tag
            transition_counts[(previous, END)] += 1
            outgoing_counts[previous] += 1

        # Rare words simulate the OOV forms encountered at evaluation time.  Keeping
        # their shape separate prevents numeric and capitalized words from diluting
        # the ordinary lowercase suffix model.
        for sentence in materialized:
            for position, (word, tag) in enumerate(sentence):
                if word_frequency[word] <= RARE_WORD_LIMIT:
                    shape = word_shape(word, position)
                    shape_tag_counts[(shape, tag)] += 1
                    shape_totals[shape] += 1
                    lower = word.lower()
                    for length in range(1, min(MAX_SUFFIX_LENGTH, len(lower)) + 1):
                        suffix = lower[-length:]
                        suffix_counts[(shape, suffix, tag)] += 1
                        suffix_totals[(shape, suffix)] += 1

        model = {
            "version": 4,
            "tags": sorted(tag_counts),
            "tag_counts": dict(tag_counts),
            "transition_counts": {f"{a}\t{b}": n for (a, b), n in transition_counts.items()},
            "outgoing_counts": dict(outgoing_counts),
            "emission_counts": {f"{tag}\t{word}": n for (tag, word), n in emission_counts.items()},
            "word_tags": {word: sorted(tags) for word, tags in word_tags.items()},
            "suffix_counts": {f"{shape}\t{suffix}\t{tag}": n for (shape, suffix, tag), n in suffix_counts.items()},
            "suffix_totals": {f"{shape}\t{suffix}": n for (shape, suffix), n in suffix_totals.items()},
            "shape_tag_counts": {f"{shape}\t{tag}": n for (shape, tag), n in shape_tag_counts.items()},
            "shape_totals": dict(shape_totals),
            "tag_context_counts": {f"{previous}\t{next}\t{current}": n for (previous, next, current), n in tag_context_counts.items()},
            "tag_context_totals": {f"{previous}\t{next}": n for (previous, next), n in tag_context_totals.items()},
            "token_count": sum(tag_counts.values()),
            "transition_alpha": TRANSITION_ALPHA,
            "emission_alpha": EMISSION_ALPHA,
            "suffix_alpha": SUFFIX_ALPHA,
            "min_suffix_observations": MIN_SUFFIX_OBSERVATIONS,
            "suffix_backoff_strength": SUFFIX_BACKOFF_STRENGTH,
            "oov_tag_context_weight": OOV_TAG_CONTEXT_WEIGHT,
            "oov_context_margin": OOV_CONTEXT_MARGIN,
        }
        return cls(model)

    def _transition_logprob(self, previous: str, current: str) -> float:
        counts = self.model["transition_counts"]
        alpha = self.model["transition_alpha"]
        denominator = self.model["outgoing_counts"].get(previous, 0) + alpha * (len(self.tags) + 1)
        return math.log((counts.get(f"{previous}\t{current}", 0) + alpha) / denominator)

    def _emission_logprob(self, tag: str, word: str) -> float:
        alpha = self.model["emission_alpha"]
        vocab_size = len(self.vocabulary) + 1  # reserve one bucket for an unseen word
        count = self.model["emission_counts"].get(f"{tag}\t{word}", 0)
        denominator = self.model["tag_counts"][tag] + alpha * vocab_size
        return math.log((count + alpha) / denominator)

    def _unknown_distribution(self, word: str, position: int) -> dict[str, float]:
        """Estimate P(tag | shape, suffixes) by smoothing from short to long suffixes."""
        shape = word_shape(word, position)
        shape_totals = self.model["shape_totals"]
        shape_tag_counts = self.model["shape_tag_counts"]
        # A fallback to LOWER makes unusual word shapes safe even when that shape
        # was not represented among rare training words.
        if shape_totals.get(shape, 0) == 0:
            shape = "LOWER"
        alpha = self.model["suffix_alpha"]
        shape_total = shape_totals.get(shape, 0)
        if shape_total:
            distribution = {
                tag: (shape_tag_counts.get(f"{shape}\t{tag}", 0) + alpha)
                / (shape_total + alpha * len(self.tags))
                for tag in self.tags
            }
        else:
            distribution = {
                tag: self.model["tag_counts"][tag] / self.model["token_count"]
                for tag in self.tags
            }

        lower = word.lower()
        suffix_totals = self.model["suffix_totals"]
        suffix_counts = self.model["suffix_counts"]
        for length in range(1, min(MAX_SUFFIX_LENGTH, len(lower)) + 1):
            suffix = lower[-length:]
            total = suffix_totals.get(f"{shape}\t{suffix}", 0)
            if total >= self.model["min_suffix_observations"]:
                suffix_distribution = {
                    tag: (suffix_counts.get(f"{shape}\t{suffix}\t{tag}", 0) + alpha)
                    / (total + alpha * len(self.tags))
                    for tag in self.tags
                }
                # Small suffix samples only gently adjust shorter-suffix evidence.
                weight = total / (total + self.model["suffix_backoff_strength"])
                distribution = {
                    tag: (1.0 - weight) * distribution[tag] + weight * suffix_distribution[tag]
                    for tag in self.tags
                }
        return distribution

    def _unknown_logprob(self, tag: str, word: str, position: int) -> float:
        return math.log(self._unknown_distribution(word, position)[tag])

    def _tag_context_logprob(self, previous: str, current: str, following: str) -> float | None:
        """Return P(current tag | previous tag, following tag), if observed."""
        totals = self.model["tag_context_totals"]
        total = totals.get(f"{previous}\t{following}", 0)
        if not total:
            return None
        counts = self.model["tag_context_counts"]
        alpha = self.model["suffix_alpha"]
        return math.log((counts.get(f"{previous}\t{following}\t{current}", 0) + alpha) / (total + alpha * len(self.tags)))

    def _viterbi(self, words: list[str]) -> list[str]:
        if not words:
            return []
        candidates = [self.model["word_tags"].get(word, self.tags) for word in words]
        scores: dict[str, float] = {}
        backpointers: list[dict[str, str]] = []
        first_word = words[0]
        for tag in candidates[0]:
            emission = self._emission_logprob(tag, first_word) if first_word in self.vocabulary else self._unknown_logprob(tag, first_word, 0)
            scores[tag] = self._transition_logprob(START, tag) + emission

        for position, (word, current_tags) in enumerate(zip(words[1:], candidates[1:]), 1):
            next_scores: dict[str, float] = {}
            pointers: dict[str, str] = {}
            for current in current_tags:
                previous = max(scores, key=lambda tag: scores[tag] + self._transition_logprob(tag, current))
                emission = self._emission_logprob(current, word) if word in self.vocabulary else self._unknown_logprob(current, word, position)
                next_scores[current] = scores[previous] + self._transition_logprob(previous, current) + emission
                pointers[current] = previous
            backpointers.append(pointers)
            scores = next_scores

        last = max(scores, key=lambda tag: scores[tag] + self._transition_logprob(tag, END))
        result = [last]
        for pointers in reversed(backpointers):
            last = pointers[last]
            result.append(last)
        return list(reversed(result))

    def _rescore_oov(self, words: list[str], first_pass_tags: list[str]) -> list[str]:
        """Use frozen first-pass neighbours to make one conservative OOV revision."""
        revised = list(first_pass_tags)
        for position, word in enumerate(words):
            if word in self.vocabulary:
                continue
            previous_tag = first_pass_tags[position - 1] if position else START
            next_tag = first_pass_tags[position + 1] if position + 1 < len(words) else END
            scores: dict[str, float] = {}
            for tag in self.tags:
                score = (
                    self._unknown_logprob(tag, word, position)
                    + self._transition_logprob(previous_tag, tag)
                    + self._transition_logprob(tag, next_tag)
                )
                context_score = self._tag_context_logprob(previous_tag, tag, next_tag)
                if context_score is not None:
                    score += self.model["oov_tag_context_weight"] * context_score
                scores[tag] = score
            original = first_pass_tags[position]
            best = max(scores, key=scores.get)
            if scores[best] > scores[original] + self.model["oov_context_margin"]:
                revised[position] = best
        return revised

    def tag(self, words: list[str]) -> list[str]:
        return self._viterbi(words)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.model, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "HMMTagger":
        return cls(json.loads(path.read_text(encoding="utf-8")))
