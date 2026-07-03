# Lung Prevention Signal Studio

A colourful Streamlit portfolio app inspired by the Cell paper **“Plasma signals of lung tumor promotion for molecular cancer prevention”**.
<img width="1663" height="946" alt="Lung Prevention Signal Studio" src="https://github.com/user-attachments/assets/b9d6679e-6b52-484a-b05e-c0277e9266b3" />
The app uses synthetic data only and includes expanded modules:

1. Cohort atlas with risk tiers and environmental burden.
2. Model performance with ROC, precision-recall, calibration, permutation importance and model-family comparison.
3. 14-protein biology heatmap and correlation module.
4. Single-cell KAC atlas simulator.
5. Longitudinal pre-diagnostic plasma signal module.
6. PM2.5 and IL-1β challenge simulator with blockade scenario.
7. Prevention trial designer with risk threshold, NNT and expected prevented cases.
8. External validation forest-plot module.
9. Interactive single-person simulator.
10. Export module for synthetic CSV and markdown summary.
11. Appearance controls with Dark mode, Light mode, colour-palette selection, continuous colour-scale selection, transparent plot panels and compact metric cards.

This is an educational and portfolio app, not a clinical device.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## One-command setup

```bash
chmod +x run_app.sh
./run_app.sh
```

