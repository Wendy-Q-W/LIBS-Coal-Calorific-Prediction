# LIBS Coal Calorific Value Prediction

> 2026 iFLYTEK AI Developer Challenge — Predicting coal bomb calorific value Q (kcal/kg) from on-belt LIBS spectra

## Overview

- **Metric**: RMSE (lower is better)
- **Training set**: Non-December data, 5 coal types, 70 batches, 944 spectra
- **Test set**: December full-month data, 5 coal types, 26 batches, 358 spectra
- **Spectral dimension**: 7,305 wavelength points, range 195.9 nm – 813.3 nm
- **Auxiliary lab indicators**: Total moisture, analytical moisture, ash, hydrogen, sulfur (labels available only in training set)
- **Core challenge**: "Many-to-one" weak alignment + train/test temporal distribution shift

### Coal Type Distribution

| Coal Type | Train Batches | Train Spectra | Test Batches | Test Spectra | CV Strategy |
|-----------|--------------|--------------|-------------|-------------|-------------|
| Zhaogu-1 (coking fines) | 11 | 140 | 2 | 28 | LOOCV |
| Zhaogu-2 (mid-coal) | 7 | 86 | 4 | 49 | LOOCV |
| Zhongma (mid-coal) | 7 | 96 | 5 | 70 | LOOCV |
| Jiulishan (mid-coal) | 18 | 262 | 6 | 93 | 100-seed GroupKFold |
| Blended coal | 27 | 360 | 9 | 118 | 100-seed GroupKFold |

## Best Results

| Version | Strategy | Online RMSE | Date |
|---------|----------|------------|------|
| **V601** | **V577 + Zhaogu-2 → V63** | **192.54** | 2026-08-26 |
| V577 | V96 + Zhongma → V72 + Blended → V63 + Zhaogu-2 → V72 | 194.45 | 2026-08-24 |
| V96 | V74+V72 blend 75/25 | 196.98 | 2026-08-14 |
| V61 | alpha=0.05 + linear bias correction | 203.21 | 2026-08-06 |

Total improvement: 203.21 → 192.54 = **-10.67**

## Model Architecture

### Core Pipeline (the only reliable one)

```
Raw spectra (7,305 dims)
    |
    +-- SNV normalization -> StandardScaler -> PCA (<=30 dims)
    |
    +-- Total intensity normalization -> Hand-crafted features (43 dims)
        |-- Statistical features (17 dims): mean/var/skew/kurt/entropy/quantiles/derivatives
        |-- Spectral line integrals (11 dims): C/H/O/N/Ca/Ca2/Mg/Al/Si/Fe/Na
        |-- Relative integrals (11 dims): line integrals / total intensity
        +-- Physical ratios (4 dims): combustible/ash, H/ash, C/ash, H/C
    |
    v
    Stage 1: Ridge x 4 -> predict auxiliary indicators OOF (moisture/ash/H/S)
    |
    Stage 2: Ridge -> [PCA + hand-crafted features + predicted auxiliaries] -> calorific value Q
    |
    Per-coal-type independent training
    Large samples: 100-seed GroupKFold ensemble
    Small samples: LOOCV deterministic
    Time weighting: Jiulishan tau=20, Blended tau=30
    Mean shrinkage: OOF-searched optimal w
    |
    Linear bias correction y = a*pred + b (V61: alpha=0.05, linear)
    Isotonic calibration (V63: alpha=0.10, isotonic; V72: alpha=0.05, isotonic)
    |
    100-seed ensemble -> batch aggregation (median) -> shrinkage -> calibration
```

### Three Baseline Models

| Model | alpha | Calibration | Online RMSE | gap |
|-------|-------|-------------|------------|-----|
| V61 | 0.05 | linear | 203.21 | ~20 |
| V63 | 0.10 | isotonic | ~205 | ~25 |
| V72 | 0.05 | isotonic | ~205 | ~20 |

### V96 Blend Composition

```
V74 = 0.5*V61 + 0.5*V63
V96 = 0.75*V74 + 0.25*V72 = 0.375*V61 + 0.375*V63 + 0.25*V72
```

### V601 Composition (Current Best, 192.54)

```
V601 = V96 base + per-coal single-model replacement:
  Zhongma      (5 batches) -> V72
  Blended coal (9 batches) -> V63
  Zhaogu-2     (4 batches) -> V63    <- Key breakthrough: V63 is 1.91 better than V72 online
  Zhaogu-1     (2 batches) -> V96 (unchanged)
  Jiulishan    (6 batches) -> V96 (unchanged)
```

## Per-Coal Online Decomposition

Starting from V96 = 196.98 baseline, the optimal model and online delta for each coal type:

| Coal Type | Best Model | Online Delta | Batches | Status |
|-----------|-----------|-------------|---------|--------|
| Zhaogu-2 | V63 | -2.60 | 4 | ✅ Largest single-coal improvement |
| Blended coal | V63 | -1.22 | 9 | ✅ |
| Zhongma | V72 | -0.61 | 5 | ✅ |
| Zhaogu-1 | V96 (unchanged) | — | 2 | V63=+1.77, V72=+0.41 both harmful |
| Jiulishan | V96 (unchanged) | — | 6 | V63=+1.44, V72=+0.72 both harmful |

**Key insight**: Models with nearly identical OOF (Zhaogu-2: V63=78, V72=75) can differ by 1.91 points online. The gap is coal × model specific and cannot be predicted from OOF.

### Per-Coal OOF RMSE (10-seed validation)

| Coal Type | V61 | V63 | V72 | V96 |
|-----------|-----|-----|-----|-----|
| Zhongma | 265 | 197 | 197 | 217 |
| Jiulishan | 117 | 51 | 48 | 70 |
| Zhaogu-1 | 75 | 31 | 36 | 44 |
| Zhaogu-2 | 108 | 78 | 75 | 85 |
| Blended coal | 155 | 110 | 114 | 121 |

## Full Iteration History

### Online Score Ranking (Top 20)

| Rank | Version | Online RMSE | Key Change |
|------|---------|------------|-----------|
| 1 | **V601** | **192.54** | V577 + Zhaogu-2 → V63 (replacing V72) |
| 2 | V577 | 194.45 | V96 + Zhongma → V72 + Blended → V63 + Zhaogu-2 → V72 |
| 3 | V592 | 194.86 | V577 + Zhaogu-1 → V72 (harmful +0.41) |
| 4 | V562 | 195.14 | V538 + Blended → V63 |
| 5 | V586 | 195.76 | V96 + Blended → V63 only |
| 6 | V566 | 196.23 | V538 + 3 coal replacements (no Jiulishan) |
| 7 | V96 | 196.98 | V74+V72 blend 75/25 |
| 8 | V189 | 196.99 | blend |
| 9 | V548 | 196.95 | All-coal optimal replacement |
| 10 | V546 | 197.09 | Zhongma + Jiulishan → V72 |
| 11 | V538 | 196.37 | V96 + Zhongma → V72 only |
| 12 | V555 | 196.67 | V96+V538 80/20 |
| 13 | V84 | 197.32 | V61+V63 55/45 |
| 14 | V83 | 197.36 | V61+V63 45/55 |
| 15 | V82 | 197.25 | V74+V72 50/50 |
| 16 | V85 | 196.99 | V74+V72 70/30 |
| 17 | V80 | 199.66 | per-coal blend |
| 18 | V92 | 206.98 | V63+V86 50/50 |
| 19 | V288 | 204.53 | SG smoothing (CV gain offset by gap) |
| 20 | V61 | 203.21 | alpha=0.05 + linear bias correction |

### Key Milestones

| Version | Online RMSE | Date | Breakthrough |
|---------|------------|------|-------------|
| V17 | 263.37 | 07-30 | Ridge with fixed alpha=0.1 replacing RidgeCV |
| V19 | 250.64 | 07-31 | PCA dimensionality reduction + key element spectral windows |
| V28 | 240.40 | 08-03 | Time weighting (Jiulishan tau=20, Blended tau=30) |
| V39 | 230.06 | 08-04 | SNV hybrid normalization |
| V55 | 209.05 | 08-05 | Linear bias correction y=a*pred+b (**-21 pts**) |
| V61 | 203.21 | 08-06 | alpha=0.05 + bias correction |
| V63 | ~205 | 08-07 | alpha=0.10 + isotonic calibration |
| V72 | ~205 | 08-08 | alpha=0.05 + isotonic calibration |
| V96 | 196.98 | 08-14 | V74+V72 blend 75/25 (**-6 pts**) |
| V538 | 196.37 | 08-22 | Zhongma → V72 (**broke 197 barrier**) |
| V562 | 195.14 | 08-23 | + Blended → V63 |
| V577 | 194.45 | 08-24 | + Zhaogu-2 → V72 |
| V601 | 192.54 | 08-26 | Zhaogu-2 → V63 (replacing V72) (**-1.91**) |

### Iteration Phase Summary

#### Phase 1: Pipeline Establishment (V17–V61, 263→203)

- V17–V28: Ridge + PCA + time weighting → 240
- V39: SNV hybrid normalization → 230
- V55: Linear bias correction (largest single breakthrough -21 pts) → 209
- V61: alpha=0.05 + bias → 203

#### Phase 2: Calibration & Ensembling (V63–V96, 205→197)

- V63/V72: isotonic calibration variants
- V74–V96: multi-model blend weight scan
- V96 = 0.375*V61 + 0.375*V63 + 0.25*V72 → 196.98

#### Phase 3: Systematic Elimination (V102–V510, all failed)

- BayesianRidge (gap=110), PCA-only Ridge (gap=120), first derivative (gap=64)
- Lasso/ElasticNet blend, PLS, Tree models (GBR/RF CV>276)
- Per-coal stacking (V61 and V72 have identical OOF, meaningless)
- SG smoothing (CV gain fully offset by gap), alpha=0.03 linear/isotonic
- All non-isotonic calibration methods (gap>50), batch_mean aggregation (V470=209.80)
- Uncertainty adjustment (shrinkage toward coal-type mean, OOF RMSE +41)

#### Phase 4: Per-Coal Optimization (V536–V601, 197→192.5)

- V538: Zhongma → V72 → 196.37 (broke 197 ceiling)
- V562: + Blended → V63 → 195.14
- V577: + Zhaogu-2 → V72 → 194.45
- V601: Zhaogu-2 → V63 (replacing V72) → **192.54** (current best)

## Eliminated Paths (Experimentally Verified Failures)

### Models & Architecture

- BayesianRidge (gap=110), PCA-only Ridge (gap=120)
- Huber regression, no-two-stage architecture, mean aggregation
- Lasso/ElasticNet, PLS, Tree models (GBR/RF)
- KernelRidge / SVR / LightGBM / CNN end-to-end
- Per-coal stacking (V61 and V72 have identical OOF)

### Preprocessing & Features

- First derivative (gap=64), second derivative
- SG smoothing (CV gain fully offset by gap)
- RobustScaler (alone 241 / with bias 229, both harmful)
- Per-coal normalization, binned/key regions/raw spectra
- Additional stoichiometric ratios/spectral lines, data augmentation

### Calibration & Aggregation

- alpha=0.03 linear/isotonic (gap=38-44)
- All non-isotonic calibration methods (gap>50)
- batch_mean aggregation (V470=209.80, gap=42)
- Fixed shrinkage weights, ensemble aggregation median/trim
- Uncertainty adjustment (shrinkage toward coal-type mean)
- Quadratic bias, CORAL domain adaptation

### Per-Coal Replacement

- Zhaogu-1 → V63 (+1.77 extremely harmful)
- Zhaogu-1 → V72 (+0.41 harmful)
- Jiulishan → V63 (+1.44 harmful)
- Jiulishan → V72 (+0.72 harmful)
- Zhongma → V63+V72 50/50 (V536=198.51 harmful)
- Intra-coal mixing (V611_3 Blended 80/20 = +0.70, V614_1 Zhongma 80/20 = +0.42)

### Global Blend

- V96+V538 80/20 (V555=196.67, worse than V538)
- Adversarial validation AV weights (V47a=242.82)
- Increased alpha (0.2/0.5)

## Core Findings

1. **Ridge variance compression**: Ridge regression systematically compresses prediction variance, especially severe for small-sample coal types (Zhaogu-2 a=1.95, capturing only 51% variance)
2. **Linear bias correction**: y = a*pred + b with only 2 parameters per coal type, near-zero overfitting risk, fixes variance compression (230→209)
3. **Lower alpha is better**: Weaker regularization → less compression → less bias-correction stretching → less noise amplification (209→203)
4. **CV is unreliable**: Both TE-CV and GK-CV are inversely correlated with online scores; can only rely on leaderboard blind testing
5. **Gap-corr inverse relationship**: corr(V96) 1.000 → gap 25, corr 0.998 → gap 31-44, corr 0.978 → gap 110
6. **OOF ≠ Online**: V61 has worst OOF for Zhongma (265) but V72 is best online; Zhaogu-2 OOF V63=78/V72=75 nearly identical but online differs by 1.91
7. **Gap is coal × model specific**: Not transferable across coal types; each coal type has a different optimal model
8. **Temporal distribution shift**: Extreme distribution shift between train/test (adversarial AUC = 1.0)

## Project Structure

```
.
|-- .gitignore                              Git ignore rules
|-- README.md                               This file
|-- 赛题概要.txt                              Official contest description
|-- 1.Baseline 哪些地方可以改进？.txt          AI-assisted analysis
|-- 2.有点门槛，但值得做.txt                   AI-assisted analysis
|-- 3.进阶.txt                               AI-assisted analysis
|-- 4.改进经验分享.txt                         External experience reference
|
|-- LIBS_project/
    |-- LIBS/
    |   |-- all_snv.py                      V39 main pipeline (SNV hybrid baseline)
    |   |-- all.py                          V17 legacy main pipeline
    |   |-- config.py                       Early modular config
    |   |-- exp_quick.py                    Quick experiment framework
    |   |
    |   |-- exp_v15*.py ~ exp_v27.py        V15-V27 early experiments
    |   |-- exp_v28.py                      V28 time weighting (online 240.40)
    |   |-- exp_v30*.py ~ exp_v38.py        V30-V38 series
    |   |-- exp_v40.py ~ exp_v43.py         V40-V43
    |   |-- exp_v45_adversarial.py          V45 adversarial validation
    |   |-- exp_v46_nonlinear.py            V46 nonlinear models
    |   |-- exp_v47_submit.py               V47 AV weight submission
    |   |-- exp_v48_augment.py              V48 data augmentation
    |   |-- exp_v49_raw_spectra.py          V49 raw spectra
    |   |-- exp_v50_ablation.py             V50 ablation study
    |   |-- exp_v51_no_tw_and_shrink.py     V51-V52 TW/shrinkage ablation
    |   |-- exp_v53_robust_median.py         V53-V56 scaler/shrinkage/bias
    |   |-- exp_v57_v60_bias_explore.py      V57-V60 bias correction refinement
    |   |-- exp_v61_v63_alpha_bias.py        V61-V66 alpha scan (V61 current baseline)
    |   |-- exp_v67_multi_directions.py      V67-V71 five new directions
    |   |-- exp_v72_v76_advanced.py          V72-V76 advanced (V72 baseline model)
    |   |-- exp_v77_v80_orthogonal.py        V77-V80 orthogonal directions
    |   |-- exp_v86_v89_new_models.py        V86-V89 Lasso and new models
    |   |-- exp_v102_v113_sg_bayesian.py     V102-V113 SG smoothing + BayesianRidge
    |   |-- exp_v143_v150_new_directions.py  V143-V150 pipeline variants
    |   |-- exp_v162_v175_pipeline_variants.py V162-V175 fine-tuning
    |   |-- exp_v181_v195_preprocessing_variants.py V181-V195 preprocessing variants
    |   |-- exp_v206_v215_percoal_pls.py     V206-V215 per-coal PLS
    |   |-- exp_v212_v215_percoal_stacking.py V212-V215 per-coal stacking
    |   |-- exp_v216_v230_alpha_sweep.py     V216-V230 alpha sweep
    |   |-- exp_v260_v270_tree_models.py     V260-V270 tree models
    |   |-- exp_v281_v292_sgsmooth_new.py    V281-V292 SG smoothing
    |   |-- exp_v281_v300_sgsmooth_variants.py V281-V300 SG smoothing variants
    |   |-- exp_v341_v357_calibration_diversity.py V341-V357 calibration diversity
    |   |-- exp_v371_v384_smart_calib.py     V371-V384 smart calibration
    |   |-- exp_v416_v425_isotonic_alpha_scan.py V416-V425 isotonic alpha scan
    |   |-- exp_v446_v460_pipeline_tuning.py  V446-V460 pipeline tuning
    |   |-- exp_v481_v490_batch_mean_alphas.py V481-V490 batch_mean alpha
    |   |
    |   |-- gen_blends_*.py                  V181-V510 blend generation scripts (11)
    |   |-- gen_v526_v535_blends.py          V526-V535 blend
    |   |-- gen_v536_v545_percoal_blend.py   V536-V545 per-coal blend
    |   |-- gen_v546_v560_percoal_expand.py  V546-V560 per-coal expansion
    |   |-- gen_v560_v575_surgical_expand.py V560-V575 surgical expansion
    |   |-- gen_v574_v590_v562_expand.py     V574-V590 V562 expansion
    |   |-- gen_v591_v610_v577_expand.py     V591-V610 V577 expansion
    |   |-- gen_v603_v620_v577_finetune.py   V603-V620 V577 fine-tuning
    |   |-- gen_v621_v640_v601_expand.py     V621-V640 V601 expansion
    |   |
    |   |-- validate_oof_disagreement.py     OOF disagreement validation
    |   |-- adversarial_validation_results.json  Adversarial validation results
    |   |-- requirements.txt                 Python dependencies
    |   |
    |   |-- output/                          Submission output directory (562 zips)
    |   |   |-- submit_v61_a005_bias.zip          <- V61 baseline model
    |   |   |-- submit_v63_isotonic.zip            <- V63 baseline model
    |   |   |-- submit_v72_isotonic_a005.zip       <- V72 baseline model
    |   |   |-- submit_v96_blend_v74_v72_7525.zip  <- V96 blend
    |   |   |-- submit_v601_v562_zhao2_v63.zip     <- V601 current best
    |   |   +-- ...                                 Other historical submissions
    |   |
    |   |-- submit_sample/                   Submission template
    |   |   |-- submit/submit.csv            26-row template
    |   |
    |   |-- train_data/                      Training data (excluded from git, 130MB)
    |   +-- test_data/                       Test data (excluded from git)
    |
    +-- (train.zip / test.zip / submit_sample.zip excluded in .gitignore)
```

## Quick Start

### Requirements

```
Python >= 3.10
numpy >= 2.0
pandas >= 2.0
scipy >= 1.10
scikit-learn >= 1.3
openpyxl >= 3.1
```

### Installation

```bash
cd LIBS_project/LIBS
pip install -r requirements.txt
```

### Data Preparation

Download `train.zip`, `test.zip`, `submit_sample.zip` from the contest platform and extract to:

```
LIBS_project/LIBS/
|-- train_data/赛题数据划分/训练集/         5 coal type subdirectories
|-- train_data/赛题数据划分/训练集标签/     5 xlsx label files
|-- test_data/赛题数据划分/测试集/          5 coal type subdirectories
+-- submit_sample/submit/submit.csv        26-row submission template
```

### Running

```bash
cd LIBS_project/LIBS

# 1. Generate three baseline models
python exp_v61_v63_alpha_bias.py    # V61 (alpha=0.05, linear)
python exp_v72_v76_advanced.py      # V72 (alpha=0.05, isotonic)
# V63 is generated in exp_v61_v63_alpha_bias.py (alpha=0.10, isotonic)

# 2. Generate V96 blend
python gen_blends_v401_v415_median_and_finetune.py

# 3. Generate V601 (current best)
python gen_v574_v590_v562_expand.py      # Generate V562
python gen_v591_v610_v577_expand.py      # Generate V577
python gen_v603_v620_v577_finetune.py    # Generate V601
```

### Output

After running, the `output/` directory contains:
- `submit.csv` — predictions (26 rows)
- `submit_vXX_xxx.zip` — packaged submission file

## Competition Info

- **Platform**: 2026 iFLYTEK AI Developer Challenge
- **Submission limit**: 3 per day
- **Team**: 2 members (Member A: materials + AI core, Member B: industry application)
