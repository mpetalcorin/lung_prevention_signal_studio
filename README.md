# Lung Cancer Prevention Signal Studio

**Interactive application for exploring synthetic plasma protein signals of lung tumour promotion, molecular prevention biology, and trial-enrichment concepts.**
<img width="1663" height="946" alt="Lung Prevention Signal Studio" src="https://github.com/user-attachments/assets/06263bce-17cb-4e85-a5b1-da28df823573" />
> **Important disclaimer**  
> This application is a research demonstration only. All data, model outputs, predictions, patient simulations, trial estimates, risk scores, and visualisations are synthetic. They must not be used for diagnosis, screening, treatment selection, clinical prevention, medical advice, regulatory submissions, or patient management.

---

## 1. Project overview

**Lung Prevention Signal Studio** is an interactive Streamlit application inspired by the Cell paper:

Pandya T, Zagorulya M, Leung MM, et al. *Plasma signals of lung tumor promotion for molecular cancer prevention*. Cell. 2026;189:3903-3921. doi:10.1016/j.cell.2026.05.005.

The paper reports a plasma proteomic signature associated with future lung cancer risk, links the signal to lung epithelial, myeloid, inflammatory, particulate-matter, EGFR-mutant, KAC/transitional-cell-state and IL-1β biology, and explores prevention-trial stratification concepts.

This app converts those concepts into a **synthetic, visual, reproducible tool** for demonstrating:

- plasma biomarker risk modelling,
- interpretable machine learning,
- synthetic cohort simulation,
- lung epithelial and myeloid biology,
- KAC/transitional-state visualisation,
- particulate matter and IL-1β challenge simulation,
- conceptual prevention-trial enrichment,
- external-validation style visualisation,
- individual risk-factor exploration,
- exportable figures and synthetic datasets.

---

## 2. Scientific concept

The uploaded reference article describes a **14-protein plasma signature** that predicted lung cancer more than five years before clinical diagnosis and was validated across multiple cohorts. The work connects circulating plasma signals to lung tumour-promoting biology, including:

- lung epithelial and myeloid sources of signature components,
- increased signal in current smokers and people exposed to particulate matter,
- EGFR-driven lung adenocarcinoma initiation,
- convergence of epithelial lineages on a keratin 8 / claudin 4 alveolar transitional state, often described as a KAC-like state,
- IL-1β inflammatory signalling,
- conceptual identification of people who may derive greater prevention benefit from anti-IL-1β therapy.

The application does **not** reproduce the paper's data. Instead, it simulates synthetic data with biologically plausible structure to demonstrate how such a platform could be built.

---

## 3. Main app modules

### 3.1 Cohort atlas

The cohort atlas provides a visual overview of the synthetic population.

It includes:

- synthetic participant count,
- simulated incident case status,
- smoking category,
- COPD history,
- PM2.5 exposure,
- protein-signature score,
- risk-score distribution,
- PM2.5-stratified signal plots,
- survival-like risk trajectories by synthetic risk quartile.

### 3.2 Model performance

This module demonstrates machine-learning evaluation for a synthetic lung-risk classifier.

It includes:

- ROC curve,
- precision-recall curve,
- calibration curve,
- confusion-matrix style summaries,
- ROC-AUC,
- average precision,
- Brier score,
- risk-threshold exploration.

The values are intentionally synthetic and should be interpreted only as a demonstration of modelling workflow.

### 3.3 Protein biology

This section explores the synthetic behaviour of the proteins used in the simulated signature.

It includes:

- protein-contribution bar charts,
- protein family grouping,
- inflammatory signalling markers,
- epithelial secretion and shedding markers,
- surfactant-associated markers,
- extracellular matrix and protease-remodelling markers,
- correlation heatmaps,
- signature-level aggregation.

### 3.4 Single-cell KAC atlas

This module visualises a synthetic single-cell inspired atlas of lung cell states.

It includes:

- AT2-like cells,
- club-like cells,
- basal-like cells,
- myeloid-like cells,
- fibroblast-like cells,
- KAC/transitional-like states,
- UMAP-style plots,
- synthetic signature enrichment across cell states,
- conceptual convergence of epithelial lineages toward a transitional state.

### 3.5 Longitudinal signals

This module simulates pre-diagnostic plasma trajectories.

It includes:

- longitudinal curves for selected proteins,
- synthetic case-control divergence over time,
- time-to-diagnosis visualisation,
- signal acceleration close to the synthetic diagnosis window.

### 3.6 PM/IL-1β simulator

This module allows users to adjust exposure and inflammatory parameters.

It includes sliders for:

- PM2.5 exposure,
- EGFR-mutant clone burden,
- IL-1β pathway activity,
- COPD-like background risk,
- smoking-related exposure.

The module generates synthetic changes in:

- plasma signature score,
- KAC-like state expansion,
- inflammatory activation,
- risk-score output.

### 3.7 Prevention trial designer

This module demonstrates the idea of enriching prevention trials by selecting higher-risk participants.

It includes:

- risk-based subgroup selection,
- conceptual number needed to treat, NNT,
- absolute risk reduction,
- relative risk reduction,
- response-index exploration,
- subgroup-size trade-offs,
- trial-enrichment visualisation.

This module is conceptual only and does not estimate real treatment benefit.

### 3.8 External validation

This section simulates how multi-cohort validation might be displayed.

It includes:

- forest-plot style visuals,
- cohort-specific effect estimates,
- synthetic confidence intervals,
- heterogeneity-style summaries,
- cross-cohort consistency plots.

### 3.9 Patient simulator

The patient simulator allows an individual synthetic profile to be explored.

Inputs may include:

- age,
- smoking category,
- pack-years,
- COPD history,
- PM2.5 exposure,
- synthetic protein profile,
- inflammatory-state assumptions.

Outputs include:

- synthetic risk score,
- top synthetic drivers,
- modifiable-factor summary,
- simulated impact of exposure changes.

This is not a clinical calculator.

### 3.10 Export

This section allows export of synthetic data and figures.

Potential exports include:

- synthetic cohort CSV,
- synthetic protein matrix,
- model-performance summary,
- visualisation-ready datasets,
- app screenshots,
- portfolio figure panels.

---

## 4. Appearance options

The app includes sidebar appearance controls:

- **Dark mode / light mode toggle**,
- categorical palette selector,
- continuous colour-scale selector,
- transparent plot-panel toggle,
- compact metric-card toggle.

The light-mode option is useful for screenshots, presentations, manuscripts, figures, and visuals.

---

## 5. Repository structure

A typical project structure is:

```text
lung_prevention_signal_studio/
├── app.py
├── README.md
├── requirements.txt
├── run_app.sh
├── push_to_github.sh
├── assets/
│   └── app_hero_white_background.png
├── data/
│   └── synthetic_exports/
└── outputs/
    └── figures/
```

The `assets`, `data`, or `outputs` folders may be generated after the app runs or after exports are created.

---

## 6. Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

---

## 7. Dependencies

Core dependencies include:

```text
streamlit
pandas
numpy
plotly
scikit-learn
```

Optional extensions may include:

```text
scipy
matplotlib
openpyxl
```

Use the provided `requirements.txt` as the authoritative dependency list for the packaged app.

---

## 8. Running the app

Open the URL:

```text
https://lungpreventionsignalstudio-iqnfkju6uprm4vqhuzesnm.streamlit.app/
```

---

## 12. Figure caption

**Lung Prevention Signal Studio.** A synthetic, interactive research dashboard for exploring plasma protein signals of lung tumour promotion, lung epithelial transitional-state biology, PM2.5 and IL-1β exposure effects, machine-learning risk stratification, and conceptual prevention-trial enrichment. The app is for demonstration only and does not use patient-level clinical data.

---

## 13. Data-generation philosophy

The application uses synthetic data designed to reflect qualitative biological relationships described in the reference paper, such as:

- higher plasma signature scores in synthetic incident cases,
- increased signal with higher PM2.5 exposure,
- stronger signal under IL-1β-like inflammatory activation,
- KAC-like state expansion under tumour-promoting conditions,
- improved conceptual trial efficiency when selecting higher-risk subgroups.

The synthetic values are not real measurements and are not calibrated for clinical use.

---

## 14. Machine-learning workflow demonstrated

The app demonstrates a practical biomedical ML workflow:

1. Generate a synthetic cohort.
2. Simulate clinical and exposure covariates.
3. Simulate plasma-protein features.
4. Aggregate a protein-signature score.
5. Train or emulate an interpretable risk model.
6. Evaluate discrimination and calibration.
7. Explore subgroup enrichment.
8. Visualise biological mechanisms.
9. Export synthetic results.

---

## 15. Model interpretation

The model-interpretation visuals are designed to show how a translational biomarker dashboard can make complex models understandable.

Examples include:

- feature-contribution bar plots,
- risk-score distributions,
- protein-correlation heatmaps,
- exposure-response plots,
- individual synthetic driver summaries.

These are demonstration outputs, not biological discovery claims.

---

## 16. Known limitations

- The app uses synthetic data only.
- It does not reproduce the original study's cohorts.
- It does not contain protected health information.
- It is not externally validated.
- It does not provide clinical prediction.
- The prevention-trial module is conceptual.
- Protein values, hazard-like effects, odds-ratio-like estimates, and NNT-like estimates are simulated.
- The visualisations are intended for research communication and demonstration.

---

## 17. Troubleshooting


### Virtual environment issues

Reset the environment:

```bash
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

---

## 18. Ethical and clinical-use disclaimer

This software is not a medical device. It does not perform diagnosis, screening, treatment selection, prevention prescribing, or real-world risk prediction. The app should not be used with real patient data unless appropriate governance, ethics approval, validation, security review, privacy review, and clinical oversight are in place.

---

## 19. Citation

Reference article:

Pandya T, Zagorulya M, Leung MM, Augustine M, Liu LY, Leppä AM, Baruchel U, Ng SW, Klockner T, Mugabo M, et al. Plasma signals of lung tumor promotion for molecular cancer prevention. Cell. 2026;189:3903-3921. doi:10.1016/j.cell.2026.05.005.

---

## 20. App Author

**Mark I. R. Petalcorin** (2026). Lung Cancer Prevention Signal Studio. https://lungpreventionsignalstudio-iqnfkju6uprm4vqhuzesnm.streamlit.app/ 

---

## 21. License

MIT 

---

This app is provided for educational, research-demonstration, and non-clinical use. 

