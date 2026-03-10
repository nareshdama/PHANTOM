# Reproducing PHANTOM Phase 1 Results

## Environment
Python 3.11 | Windows 10/11 | 32GB RAM

## Steps
1. `git clone https://github.com/nareshdama/PHANTOM.git && cd PHANTOM`
2. `python -m venv venv && venv\Scripts\activate`
3. `pip install -r requirements.txt`
4. `python experiments/baseline_validation.py --seed 42`
5. `python experiments/ramp_rate_sweep.py --seed 42 --runs 200`
6. `python analysis/generate_report.py`
7. Open `docs/PHANTOM_Phase1_ValidationReport.md`

## Expected Results
- Baseline mean miss: `< 5.0 m`
- Critical rate: `~0.035 rad/s` in the current validated Phase 1 build
- Mean miss at critical rate: `> 200 m`
- Detection rate: `< 5%`
