# POS Tagging using a Bigram HMM

A Python 3 implementation of a Part-of-Speech (POS) tagger based on a
first-order Bigram Hidden Markov Model (HMM).

The system uses transition and emission probabilities with additive smoothing,
Viterbi decoding, and a suffix/word-shape based approach for handling
out-of-vocabulary (OOV) words.

## Results

The final model achieves:

| Metric | Accuracy |
|---|---:|
| **Overall** | **96.4080%** |
| Known words | 96.70% |
| OOV words | 85.44% |

Development set:

- Total tokens: **47,633**
- Known tokens: **46,417**
- OOV tokens: **1,216**
- Correctly tagged: **45,922**

## Model

The tagger is a first-order HMM with:

- **Transition probabilities:** \(P(t_i \mid t_{i-1})\)
- **Emission probabilities:** \(P(w_i \mid t_i)\)
- **Viterbi decoding** for finding the most likely tag sequence
- **Additive smoothing** for unseen transitions and emissions
- **Suffix and word-shape features** for OOV words
- **Second-pass OOV rescoring**

Probabilities are computed in log space during decoding to avoid numerical
underflow.

## Final Configuration

| Parameter | Value |
|---|---:|
| Transition smoothing \(\alpha_T\) | **0.20** |
| Emission smoothing \(\alpha_E\) | **0.005** |
| Rare-word limit | **4** |
| Maximum suffix length | **5** |
| Minimum suffix observations | **2** |
| Suffix backoff strength | **20** |

These values were selected through experiments on the development set.

## Repository Structure

```text
NLP-PA1/
│
├── data/
│   ├── train.tags
│   ├── dev.sents
│   └── dev.tags
│
├── src/
│   └── pos_tagger/
│       └── hmm.py
│
├── scripts/
│   ├── evaluate.py
│   └── error_analysis.py
│
├── notebooks/
│   └── analysis.ipynb
│
├── figures/
│   ├── known_vs_oov_accuracy.pdf
│   └── confusion_matrix.pdf
│
├── tests/
│   └── ...
│
├── results/
│   ├── model.json
│   └── dev.out
│
├── build_tagger.py
├── run_tagger.py
└── README.md