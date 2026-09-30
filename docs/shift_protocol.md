# Study v2: missingness shift and limited calibration labels

This protocol is fixed in Git before inspecting v2 performance. It is an internal
prespecification, not an external preregistration. The earlier German Credit
matched-mechanism experiment remains reproducible with `configs/experiment.yaml`.

## Questions and outcomes

1. Does explicit mask information improve Brier score under a change from MCAR
   training to stronger value-dependent missingness?
2. Does recalibration on labeled target observations improve transferred source
   calibration, and how does this depend on 25/50/100 calibration observations?
3. How do these effects compare with a native-missing gradient-boosted tree?

Primary contrasts: mask minus value-only Brier for LR, RF and MLP, separately on
each of three datasets, at MNAR rate .5, strength 4, source Platt n=100. These
nine contrasts are reported together regardless of outcome. Other conditions,
calibration contrasts, AUC, F1, ECE and clipped log loss are secondary/descriptive.
No test-set hyperparameter selection, winner selection, or significance test.

## Data and partitions

Use the full official UCI German Credit (1,000), Credit Approval (690), and
Default of Credit Card Clients (30,000) files. Pin raw-file SHA-256; retain natural
NaNs and all rows. Approval uses 1=approved, German uses 1=Bad, Taiwan uses 1=next
month default. These are different outcomes, not interchangeable default risks.
Drop Taiwan ID. Keep documented unknown/other categorical codes as observed.
Category transport codes come from fixed schemas; model encoders fit on train.

Group identical original feature rows, regardless of labels. Use shuffled
StratifiedGroupKFold, five outer folds with seed 2026. Within each outer training
pool, take the first split from four-fold StratifiedGroupKFold with seed
2026+outer_fold; its held-out groups form calibration, the rest training.
Approximate proportions: 60% training, 20% calibration, 20% test. Each original
row appears once in out-of-fold evaluation per condition. Preserve all row IDs,
groups, class counts, fingerprints, and actual partition sizes. Calibration
budgets are nested class-proportion-balanced random prefixes; they are not
independent replicates. Never rebalance to 50:50. Both classes must occur.

## Simulation and models

All base models train once per dataset/fold on MCAR .3. Reuse them for 13 target
conditions in `configs/shift_study.yaml`; thresholds use original training rows
only. Mask sampling uses split-specific deterministic seeds, identical across
models. Rates mean the fraction of initially observed eligible cells deleted;
existing missingness is retained. MAR drivers remain observed. MNAR strength 4
gives rank-based sampling weights from 1 to 5 (not a fivefold deletion odds ratio).
Artificial missingness does not identify the mechanism of the original NaNs.

LR and RF pairs use the same train-only imputation/one-hot/scaling and append
constant ones versus observed masks. RF considers all features at every split
to avoid changing the candidate-feature fraction when indicators are appended;
both variants have identical input dimensions. MLP uses the existing matched
constant-channel control. Fixed configurations are methodological controls, not
equal-cost optimized state-of-the-art comparisons. LightGBM uses native NaNs;
nominal fields have train-derived categorical support, unknown test categories
become missing. No class weighting, early stopping, tuning, or test refitting.

Source calibrators use MCAR .3 calibration rows and transfer unchanged. Target
calibrators use the *same row identities and labels*, re-masked under the target
condition. This is controlled supervised adaptation, requiring labeled target
examples; it is not label-free domain adaptation. For the matched MCAR .3
condition, source and target inputs and predictions must coincide exactly.
Each mode uses nested budgets 25/50/100, Platt on clipped logits and isotonic.
Retain uncalibrated results once. Test rows never fit any component.

## Analysis and reproducibility

Compute pooled OOF metrics within each dataset, never pool different labels.
Primary Brier differences use paired loss differences on the same rows. Draw
2,000 bootstrap samples of original-feature groups within each outer test fold,
retaining all members and weighting by the resulting observation count. Report
percentile 95% intervals conditional on fitted models, sampled masks, and chosen
calibration subsets. These intervals do not capture retraining/partition/mask
uncertainty, do not make folds independent, and are not multiplicity-adjusted.
Report all nine primary intervals without dichotomous significance language.

Save compressed float64 prediction arrays per dataset/fold and JSON audit
reports outside Git. Keep generated results and figures locally. Validate disjoint groups,
one-time OOF coverage, nested calibration budgets, raw NaN/mask consistency,
train-only learned preprocessing, probability bounds, and matched-condition
source/target equivalence. Re-run a fixed fold for exact reproducibility.

Limitations are prespecified: historical benchmarks, simulated collection shift,
three distinct outcomes, limited calibration draws, fixed model budgets, and no
temporal/external clinical or financial deployment validation. Do not infer
fairness, causal effects, or universal superiority.
