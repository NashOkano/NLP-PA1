import sys
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pos_tagger.hmm import HMMTagger, word_shape


class HMMTaggerTests(unittest.TestCase):
    def setUp(self):
        self.tagger = HMMTagger.train([
            [("the", "DT"), ("book", "NN")],
            [("book", "VB"), ("now", "RB")],
            [("the", "DT"), ("dog", "NN")],
        ])

    def test_context_disambiguates_known_word(self):
        self.assertEqual(self.tagger.tag(["the", "book"]), ["DT", "NN"])

    def test_unknown_word_has_complete_tag_sequence(self):
        result = self.tagger.tag(["the", "glorping"])
        self.assertEqual(len(result), 2)
        self.assertTrue(set(result).issubset({"DT", "NN", "VB", "RB"}))

    def test_word_shapes_ignore_sentence_initial_capitalization(self):
        self.assertEqual(word_shape("London", 0), "LOWER")
        self.assertEqual(word_shape("London", 1), "CAPITALIZED")
        self.assertEqual(word_shape("NASA", 1), "ALL_CAPS")
        self.assertEqual(word_shape("30,537", 1), "HAS_DIGIT")
        self.assertEqual(word_shape("hard-charging", 1), "HYPHENATED")

    def test_unknown_distribution_is_smoothed(self):
        distribution = self.tagger._unknown_distribution("glorping", 1)
        self.assertAlmostEqual(sum(distribution.values()), 1.0)
        self.assertTrue(all(value > 0 for value in distribution.values()))

    def test_oov_rescoring_uses_a_single_frozen_pass(self):
        tags = self.tagger._viterbi(["the", "glorping"])
        self.assertEqual(len(self.tagger._rescore_oov(["the", "glorping"], tags)), 2)

    def test_saved_model_round_trips(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            self.tagger.save(path)
            self.assertEqual(HMMTagger.load(path).tag(["book", "now"]), ["VB", "RB"])


if __name__ == "__main__":
    unittest.main()
