# PHANTOM
## Proportional-Navigation Heuristic Adaptive Missile-Tracking Nullification

**Author:** Naresh Dama | Department of Aerospace and Mechanical Engineering  
**Location:** Jacksonville, Florida | March 2026  
**Target Publication:** IEEE Transactions on Aerospace and Electronic Systems (TAES)  
**Current Phase:** Phase 1 — Theoretical Foundation  

---

## What is PHANTOM?

PHANTOM is a research simulation framework that investigates kinematically 
consistent false signal injection as an electronic warfare countermeasure 
against Proportional Navigation (PN)-guided missiles equipped with Extended 
Kalman Filter (EKF) state estimators.

The core insight: the EKF's chi-squared innovation gate — designed to reject 
outlier measurements — can be exploited. By injecting a false Line-of-Sight 
(LOS) angular rate that stays just below the detection threshold, the missile 
confidently tracks a phantom trajectory to a controlled wrong intercept point.

**Key results (simulation-validated):**
- Mean miss distance: 203 ± 27 m (baseline: < 5 m)
- Detection rate: < 5% (EKF innovation gate maintained below χ² = 9.21)
- Critical injection rate: ~0.032 rad/s (linear ramp profile)

---

## Project Structure

```
phantom/
├── core/                # PhantomSimulator engine (EKF, PN, injection profiles)
├── experiments/         # Validation experiments (baseline, ramp sweep, Monte Carlo)
├── training/            # PhantomDatasetGenerator + PhantomLLMController pipeline
├── analysis/            # Statistics, visualization, report generation
├── tests/               # pytest suite (≥85% coverage required)
├── configs/             # phantom_config.yaml — single source of truth
├── data/results/        # Simulation outputs — gitignored
├── figures/             # Publication figures at 300 DPI — gitignored
├── models/              # LLM weights — gitignored
└── docs/                # Validation reports, API docs
```

## Phase Roadmap

| Phase | Goal | Duration | Status |
|-------|------|----------|--------|
| 1 | Mathematical Foundation | Months 1–2 | In Progress |
| 2 | Simulation & Experiments | Months 3–6 | Pending |
| 3 | LLM Training Pipeline | Months 7–10 | Pending |
| 4 | Jetson Edge Deployment | Months 11–14 | Pending |
| 5 | IEEE TAES Publication | Months 15–18 | Pending |
| 6 | Extensions & Commercialization | Months 19–24 | Pending |

## Quick Start

```bash
# 1. Clone and enter
git clone https://github.com/nareshdama/phantom.git
cd phantom

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run test suite
pytest tests/ -v

# 5. Run baseline experiment
python experiments/baseline_validation.py --seed 42
```

## Citation

```
@software{dama2026phantom,
  author    = {Naresh Dama},
  title     = {PHANTOM: Proportional-Navigation Heuristic Adaptive 
               Missile-Tracking Nullification},
  year      = {2026},
  publisher = {Zenodo},
  note      = {Phase 1 — IEEE TAES submission in preparation}
}
```
