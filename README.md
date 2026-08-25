# PA-1: Version 4 Bigram HMM POS tagger

This implementation preserves the required assignment interface:

```powershell
python build_tagger.py data/train.tags model.json
python run_tagger.py data/dev.sents model.json results/dev.out
```

Version 4 is a first-order (bigram) HMM decoded with log-space Viterbi. It uses additive smoothing for transitions and known-word emissions. Unknown words use distributions learned from rare training words, separated by word shape (capitalized, all-caps, digit-containing, hyphenated, or lowercase). Suffix evidence is interpolated from short to long suffixes rather than relying on one suffix match; sentence-initial capitalization is intentionally ignored. A single conservative second pass re-scores OOV words using frozen neighboring Viterbi tags and a learned distribution for the middle tag conditioned on that tag pair.

Evaluate the dev output with:

```powershell
python scripts/evaluate.py results/dev.out data/dev.tags
```

Run the basic regression tests with:

```powershell
python -m unittest discover -s tests -v
```
