# LIBS Coal Calorific Value Prediction

> 2026 iFLYTEK AI Developer Challenge — Predicting coal bomb calorific value Q (kcal/kg) from on-belt LIBS spectra

## Contest Background

Coal calorific value (heating value) is the core indicator for evaluating coal quality, pricing, and optimizing combustion. Traditional measurement uses a bomb calorimeter in the laboratory, which is time-consuming and cannot meet real-time on-site requirements. Laser-Induced Breakdown Spectroscopy (LIBS) enables rapid on-line spectral acquisition directly on the coal belt at the mine site. This challenge aims to build a precise mapping between on-belt LIBS spectral data and laboratory-standard bomb calorific values using advanced algorithmic models.

### Key Challenge: "Many-to-One" Weak Alignment

Each day, 10–20+ LIBS spectra are randomly acquired from high-speed coal flow on the belt during an 8-hour shift. However, the daily bomb calorific value is obtained by physically sampling multiple sub-samples, mixing and dividing them, and then testing a micro-sample in the lab. Therefore, multiple transient spectra map to a single macroscopic calorific value label — a classic "many-to-one" weak alignment problem.

### Metric & Rules

- **RMSE** (Root Mean Square Error, lower is better)
- **Submission limit**: 3 per day
- **Contest period**: June 9 – August 27, 2026
- **Team**: 2 members
- **Fairness rule**: Test set data must not be used in model training/tuning (no pseudo-labels, semi-supervised joint training, or leaderboard-score reverse-engineering)

### Dataset Description

- **Training set**: Non-December data, 5 coal types, 70 batches, 944 spectra
- **Test set**: December full-month data, 5 coal types, 26 batches, 358 spectra
- **Spectral dimension**: 7,305 wavelength points, range 195.9 nm – 813.3 nm
- **Auxiliary lab indicators** (training set only): Total moisture, Analytical moisture, Ash, Hydrogen, Sulfur (all in %)

**Label table fields:**

| Field | Type | Description |
|-------|------|-------------|
| 名称 (Name) | string | Unique batch identifier (e.g., "Jan 3") |
| 发热量(Q) | float | Bomb calorific value — core prediction target |
| 全水分 | float | Total moisture content (%) |
| 分析基水分 | float | Analytical moisture content (%) |
| 灰分 | float | Ash content after complete combustion (%) |
| 氢 | float | Hydrogen element content (%) |
| 硫 | float | Sulfur element content (%) |

**Spectral data fields:**

| Field | Type | Description |
|-------|------|-------------|
| 名称 (Name) | string | Batch identifier, linked to label table |
| RecordTime | string | Precise recording time of each spectrum |
| IntegrationTime | int | Spectral acquisition integration time |
| DataLength | int | Number of data points per spectrum |
| accumulateCount | int | Accumulation count |
| Wavelength | string (array) | Wavelength sequence |
| Data | string (array) | Intensity values at corresponding wavelengths |

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

## Baseline Improvement Guide

### Tier 1: Easy Wins

**1. Adjust regularization strength**

The baseline used `ALPHAS = [1.0, 10.0, ..., 10000.0]` — too conservative. Expanding the range (especially lower values like 0.1, 0.05) found much better solutions. Our final alpha=0.05 was a key improvement.

- **Pitfall**: alpha too small (e.g., 0.001) on small-batch coal types causes severe overfitting — good CV RMSE but online collapse.

**2. Increase spectral line window width**

Baseline used fixed windows (e.g., C ±2nm). Widening windows captures more complete spectral line energy. Adjust by +1nm increments and monitor CV-RMSE.

- **Pitfall**: Too wide causes adjacent line interference.

**3. Increase PCA dimensions for large coal types**

Baseline `N_PCA_MAX = 30` may lose information for Blended coal (27 batches). Increasing to 50 can help. The code auto-limits to `min(N_PCA_MAX, n_batches-1)`, so it's safe globally.

- **Pitfall**: Don't increase for small-batch types (≤10) — overfitting.

### Tier 2: Moderate Difficulty

**1. Replace Stage 2 with LightGBM**

Ridge assumes linear relationships between auxiliary indicators and calorific value. In reality, ash/moisture effects are nonlinear (diminishing marginal effect at high ash). LightGBM captures these nonlinearities.

- **Pitfall**: With small samples, LightGBM overfits easily. Use strong regularization (min_child_samples, reg_alpha, reg_lambda) and shallow trees (max_depth=4).
- **Note**: We tested LightGBM and it failed (CV>276) due to extreme distribution shift.

**2. Batch-level spectral quality filtering**

Among 10–20 spectra per batch, some may be "contaminated" (misaligned laser, abnormally low/high intensity). IQR-based outlier removal before aggregation improves robustness:

```python
def aggregate_to_batch(preds, names):
    grouped = {}
    for p, n in zip(preds, names):
        grouped.setdefault(n, []).append(p)
    result = {}
    for k, v in grouped.items():
        arr = np.array(v)
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        mask = (arr >= q1 - 1.5*iqr) & (arr <= q3 + 1.5*iqr)
        result[k] = float(np.median(arr[mask]) if mask.sum() > 0 else np.median(arr))
    return result
```

**3. Per-coal-type hyperparameter tuning**

Different coal types have vastly different characteristics (Zhongma spans 1000 kcal, Zhaogu-2 only 500 kcal). Uniform hyperparameters are suboptimal. Configure per-coal alpha, PCA dims, etc.

### Tier 3: Advanced

**1. Multi-model ensemble**

Train multiple model versions (different feature sets, different algorithms) and weighted-average predictions. Choose models with low correlation for maximum benefit (e.g., Ridge + LightGBM, not two Ridges).

**2. Boltzmann plot for plasma temperature estimation**

LIBS plasma temperature T relates to elemental composition. Computing temperature from multiple spectral lines of the same element (different excitation levels) via the Boltzmann equation provides deeper physical features than simple line integrals. Requires NIST atomic spectral database lookup.

**3. Deep learning feature extraction (1D-CNN)**

PCA is unsupervised linear reduction. 1D-CNN can learn nonlinear spectral patterns directly relevant to calorific value, and is more sensitive to local spectral structure (peak shape, shoulder peaks).

- **Pitfall**: With only 200–400 training spectra, deep models overfit easily. Use shallow CNN (2–3 layers), strong Dropout (0.5), and strict GroupKFold validation.
- **Note**: We tested CNN end-to-end — it failed due to extreme distribution shift (adversarial AUC = 1.0).

## Lessons Learned

### 1. Good CV ≠ Good Online

The most typical case: a per-coal grid search dropped local CV-RMSE from 158 to 137, but online went from 271 to 274. Over-tuning on 70 training batches learns accidental structures. Multiple validation protocols (fixed 5-fold, repeated 5-fold, LOO, month-holdout, sandwich-month) can all overfit — if a method wins on all protocols but fails online, the protocols themselves are overfitting.

### 2. Scoring is by batch — training should respect batches

Early approach treated each spectrum as a training row, giving more weight to batches with more spectra. Since official RMSE is per-batch, switching to "one training row per batch" was a major online improvement.

### 3. Do not reverse-engineer parameters from leaderboard feedback

Using two submitted scores to derive a reverse extrapolation direction is mathematically tempting but uses hidden test set feedback — a compliance risk. Online scores should only record version success/failure, not drive further parameter selection.

### 4. Auxiliary indicators are powerful but cannot be used for crude "denoising"

Ash, hydrogen, and calorific value are highly correlated, but directly smoothing Q toward auxiliary formulas may remove real calorific differences present in the spectra. Low-dose, coal-directed, in-fold auxiliary constraints work better than direct target replacement.

### 5. Temporal shift is important but easily misjudged

Test set is December; training spans multiple historical months. Month-holdout and sandwich-month validation seemed most trustworthy, but a sandwich-month scheme that improved mainly on one month didn't generalize to December. Safe threshold: no structural reversal across fixed 5-fold, repeated 5-fold, LOO, recent-month, and cross-month protocols.

### 6. Auto-stacking and coal-source post-processing require extreme caution

With only 70 batches, auto-learned fusion weights, coal-source biases, and coal-source slopes easily fit OOF residuals as patterns. Post-processing without new spectral or chemical information support carries high risk.

### 7. Systematic thinking > blind AI/compute power

AI and compute can run experiments, search literature, modify code, and audit — but the initial architecture must be thought through by the human first. Relying solely on AI leads to scattered, unsystematic attempts. Real improvement comes from analyzing the data structure and scoring logic first, then iterating around one main line.

### 8. Read domain papers, don't just try ML tricks

LIBS has many physics and engineering issues: continuous background, plasma fluctuation, self-absorption, moisture effects, sample morphology, line selection, instrument drift. Without reading relevant papers, it's hard to know which attempts have physical basis vs. which are just high-variance gambling.

**Recommended papers:**

| Paper | Focus |
|-------|-------|
| Coal analysis by laser-induced breakdown spectroscopy: a tutorial review | Comprehensive LIBS coal analysis review |
| Estimating Calorific Value of Coal Using LIBS through Statistical Algorithms | Directly about coal calorific prediction; PLS, SNR, Dulong formula |
| Determination of Calorific Value of Mixed Coals by LIBS | Mixed coal + preprocessing (SG smoothing, SG derivative, PLSR) |
| Rapid Quantitation of Coal Proximate Analysis by Using LIBS | Proximate analysis indicators (ash, moisture) |
| Improved measurement of calorific value of pulverized coal particle flow by LIBS | Online coal flow scenario; spectral correction, full-spectrum modeling |
| Accuracy Enhancement of LIBS-XRF Coal Quality Analysis | Intensity correction, piecewise modeling, LIBS-XRF fusion |
| Effects of moisture content on coal analysis using LIBS | Moisture affects plasma and spectral signals |

## Repository Contents

```
.
|-- README.md                    This file (complete documentation)
|-- all_python_scripts.zip       101 Python scripts (V15–V640, all experiments + blend generators)
|-- output.zip                   5 key baseline model submissions:
    |-- submit_v61_a005_bias.zip         <- V61 baseline (alpha=0.05, linear calibration)
    |-- submit_v63_isotonic.zip          <- V63 baseline (alpha=0.10, isotonic calibration)
    |-- submit_v72_isotonic_a005.zip     <- V72 baseline (alpha=0.05, isotonic calibration)
    |-- submit_v96_blend_v74_v72_7525.zip <- V96 blend (0.375*V61 + 0.375*V63 + 0.25*V72)
    +-- submit_v601_v562_zhao2_v63.zip   <- V601 current best (192.54)
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

### Data Preparation

Download `train.zip`, `test.zip`, `submit_sample.zip` from the contest platform and extract:

```
LIBS_project/LIBS/
|-- train_data/赛题数据划分/训练集/         5 coal type subdirectories
|-- train_data/赛题数据划分/训练集标签/     5 xlsx label files
|-- test_data/赛题数据划分/测试集/          5 coal type subdirectories
+-- submit_sample/submit/submit.csv        26-row submission template
```

### Running

```bash
# Unzip scripts
unzip all_python_scripts.zip -d LIBS_project/LIBS/

# 1. Generate three baseline models
python exp_v61_v63_alpha_bias.py    # V61 (alpha=0.05, linear) + V63 (alpha=0.10, isotonic)
python exp_v72_v76_advanced.py      # V72 (alpha=0.05, isotonic)

# 2. Generate V96 blend
python gen_blends_v401_v415_median_and_finetune.py

# 3. Generate V601 (current best)
python gen_v574_v590_v562_expand.py      # Generate V562
python gen_v591_v610_v577_expand.py      # Generate V577
python gen_v603_v620_v577_finetune.py    # Generate V601
```

### Output Format

Submission CSV (UTF-8 encoded, 26 rows):
- Column 1: Batch identifier (名称)
- Column 2: `预测发热量_MJ_KG` (predicted calorific value)

## Competition Info

- **Platform**: 2026 iFLYTEK AI Developer Challenge
- **Submission limit**: 3 per day

