# Antigen — evaluation methodology

This document records how Antigen is evaluated and the exact results emitted
by the checked-in evaluation programs. The two reports below were produced by:

```bash
.venv/bin/python -m eval.harness
.venv/bin/python -m eval.robustness
```

Both commands train and score locally from the checked-in data. They make no
network calls and do not reuse a persisted model artifact.

## What is being measured

`eval.harness` creates a stratified held-out split of the generated corpus,
trains the classifier only on the training portion, and evaluates it against
two test sources:

- **Held-out test split:** examples withheld from model fitting. This measures
  performance on unseen rows, but examples can still share generator structure
  with the training portion.
- **Adversarial suite:** separately authored examples whose phrasing and
  generation path are disjoint from the training generator. This is a harder
  generalization check, but it remains a small, project-authored synthetic
  benchmark rather than independent production evidence.

The run used 241 training examples and 81 held-out examples. The adversarial
suite contained 30 examples.

The report uses these metrics:

- **Precision:** among examples assigned a class, the fraction that truly
  belongs to that class. Injection precision captures alert quality.
- **Recall:** among examples that truly belong to a class, the fraction the
  classifier finds. Injection recall captures how many attacks are detected.
- **F1:** the harmonic mean of precision and recall; useful when neither error
  type should dominate the summary.
- **Support:** the number of examples of that class in the evaluation set.
- **Accuracy:** the fraction of all examples classified correctly.
- **ROC AUC:** how well the score ranks injections above benign examples across
  thresholds. It does not select a deployment threshold.
- **False-positive rate:** `FP / (FP + TN)`, the fraction of benign examples
  flagged at a given operating point.
- **Per-category accuracy:** accuracy within each source category. This keeps a
  strong aggregate result from hiding a weak category.

The standard class metrics and confusion matrices use the default 0.5 decision
threshold. Scores in the threshold sweep include the same guarded discussion-
context cap used by normal prediction.

## Held-out results

The held-out test split contained 81 examples: 41 benign and 40 injection.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Benign | 0.974 | 0.902 | 0.937 | 41 |
| Injection | 0.907 | 0.975 | 0.940 | 40 |

| Metric | Result |
|---|---:|
| Accuracy | 0.938 |
| ROC AUC | 0.963 |

The confusion matrix `[[TN, FP], [FN, TP]]` was `[[37, 4], [1, 39]]`.
At the default operating point, that means four benign examples were flagged
and one injection was missed.

### Held-out per-category accuracy

| Category | Accuracy |
|---|---:|
| `benign` | 1.000 |
| `benign_hard` | 0.692 |
| `benign_wrapped` | 1.000 |
| `direct_injection` | 0.875 |
| `indirect_injection` | 1.000 |
| `jailbreak_roleplay` | 1.000 |
| `obfuscated_base64` | 1.000 |
| `obfuscated_hex` | 1.000 |
| `obfuscated_zero_width` | 1.000 |

`benign_hard` is the clear held-out weak spot. These examples quote, discuss,
translate, test, or report attack-like text without issuing it as a live
instruction. The surface vocabulary therefore overlaps heavily with the
positive class.

## Adversarial-suite results

The unseen-phrasing suite contained 30 examples: 10 benign and 20 injection.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Benign | 1.000 | 0.900 | 0.947 | 10 |
| Injection | 0.952 | 1.000 | 0.976 | 20 |

| Metric | Result |
|---|---:|
| Accuracy | 0.967 |
| ROC AUC | 0.990 |

The confusion matrix `[[TN, FP], [FN, TP]]` was `[[9, 1], [0, 20]]`.
At the default operating point, one benign example was flagged and no
injections were missed.

### Adversarial-suite per-category accuracy

| Category | Accuracy |
|---|---:|
| `benign` | 1.000 |
| `benign_hard` | 0.750 |
| `direct_injection` | 1.000 |
| `indirect_injection_browser_use` | 1.000 |
| `indirect_injection_coding_agent` | 1.000 |
| `indirect_injection_tool_use` | 1.000 |
| `jailbreak_roleplay` | 1.000 |
| `obfuscated` | 1.000 |

The suite is useful because its wording is not produced by the main dataset
generator. It is not a claim of broad real-world coverage: the suite has only
30 examples, was authored within this project, and does not represent an
adaptive external test campaign.

## Threshold sweep

Threshold choice determines the balance between missed attacks and false
alarms. Lower thresholds generally improve injection recall while increasing
the benign false-positive rate; higher thresholds generally do the reverse.
The appropriate point depends on whether the detector blocks content, queues
it for review, or contributes one signal to a broader policy.

### Held-out operating points

| Threshold | Injection precision | Injection recall | Injection F1 | False-positive rate |
|---:|---:|---:|---:|---:|
| 0.1 | 0.690 | 1.000 | 0.816 | 0.439 |
| 0.2 | 0.830 | 0.975 | 0.897 | 0.195 |
| 0.3 | 0.848 | 0.975 | 0.907 | 0.171 |
| 0.4 | 0.907 | 0.975 | 0.940 | 0.098 |
| 0.5 | 0.907 | 0.975 | 0.940 | 0.098 |
| 0.6 | 0.905 | 0.950 | 0.927 | 0.098 |
| 0.7 | 0.897 | 0.875 | 0.886 | 0.098 |
| 0.8 | 0.919 | 0.850 | 0.883 | 0.073 |
| 0.9 | 0.933 | 0.700 | 0.800 | 0.049 |

### Adversarial-suite operating points

| Threshold | Injection precision | Injection recall | Injection F1 | False-positive rate |
|---:|---:|---:|---:|---:|
| 0.1 | 0.800 | 1.000 | 0.889 | 0.500 |
| 0.2 | 0.952 | 1.000 | 0.976 | 0.100 |
| 0.3 | 0.952 | 1.000 | 0.976 | 0.100 |
| 0.4 | 0.952 | 1.000 | 0.976 | 0.100 |
| 0.5 | 0.952 | 1.000 | 0.976 | 0.100 |
| 0.6 | 1.000 | 0.900 | 0.947 | 0.000 |
| 0.7 | 1.000 | 0.850 | 0.919 | 0.000 |
| 0.8 | 1.000 | 0.850 | 0.919 | 0.000 |
| 0.9 | 1.000 | 0.650 | 0.788 | 0.000 |

These operating points are empirical counts on small checked-in sets. In
particular, a reported false-positive rate of 0.000 does not establish that a
deployment will produce no false positives; it means none occurred in the
relevant benchmark at that threshold.

## Model versus regex-only baseline

The baseline applies a fixed rule over the interpretable regex and structural
signals, with no learned character n-gram component. Scoring it on the same
examples tests whether the trained model adds measurable generalization beyond
literal pattern matching.

| Evaluation set | Model accuracy | Regex baseline accuracy | Absolute lift |
|---|---:|---:|---:|
| Held-out | 0.938 | 0.568 | +0.370 |
| Adversarial, unseen phrasing | 0.967 | 0.300 | +0.667 |

The larger adversarial-suite lift is consistent with the intended distinction:
the regex baseline can only react to signals its fixed patterns recognize,
while the learned character features can assign weight to related unseen
phrasing. This comparison does not show that the trained model is sufficient
for production; it shows that it outperforms this explicit baseline on these
two checked-in sets.

## Adversarial robustness

`eval.robustness` asks a different question from clean-set accuracy: among
injection examples the fully trained model catches in clear text at threshold
0.5, how many remain caught after a semantics-preserving character-level
evasion is applied?

The evaluator trains on the complete generated corpus, pools injection examples
from that corpus and the adversarial suite, then conditions each transform on
the same 171 baseline-caught attacks. Each transform is applied separately with
a fixed random sequence.

| Evasion | Baseline caught | Retained after evasion | Retention |
|---|---:|---:|---:|
| `zero_width_insertion` | 171 | 171 | 100.0% |
| `homoglyph_substitution` | 171 | 143 | 83.6% |
| `intra_word_spacing` | 171 | 167 | 97.7% |
| `case_flip` | 171 | 171 | 100.0% |
| `leetspeak` | 171 | 138 | 80.7% |

Case flipping and zero-width insertion retained every baseline-caught attack in
this run. Intra-word spacing caused a smaller drop. Homoglyph substitution and
leetspeak were materially weaker, with leetspeak the lowest-retention transform
at 80.7%.

Retention is deliberately conditional. It does not give credit for attacks the
model already missed before perturbation, does not measure benign false
positives, and does not test chained transforms, semantic paraphrase,
multilingual attacks, or an adversary adapting through repeated probes. It
isolates how much detection survives each checked-in evasion family.

## Interpretation and limits

The held-out split is useful for repeatable regression testing, but shared
generator structure can make it optimistic. The adversarial suite reduces that
specific overlap without becoming an independent benchmark. The robustness
run adds an adaptive-security dimension, but only for its listed character-
level transforms and only among baseline-caught attacks.

The most defensible reading is therefore comparative and category-specific:
the learned model adds substantial lift over the regex baseline on both test
sets; quoted or discussed attacks remain the most visible false-positive
class; and cheap character evasions vary from fully retained in this run to a
meaningful detection loss under leetspeak and homoglyph substitution. None of
the reports establishes production readiness or a universal probability of
attack.
