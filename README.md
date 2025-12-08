# Creative Fatigue Analysis in Digital Advertising

Analysis of creative fatigue in digital advertising using the Criteo Attribution Dataset.

---

## Abstract

This research investigates **creative fatigue** in digital advertising—the phenomenon where users become less responsive to ads as they see them repeatedly. Using 16.5 million advertising impressions from the Criteo Attribution Dataset, we address three research questions: (1) How does click-through rate (CTR) change with repeated ad exposure? (2) Are certain ad types more resistant to engagement decline? (3) Can we predict when an ad has "run its course"?

**Key Findings:**
- CTR declines **25-40%** by the 3rd-5th exposure when controlling for survivorship bias
- High-performing campaigns are **36% more resistant** to fatigue than low-performing campaigns
- Fatigue is predictable with **AUC of 0.70+** using early engagement signals
- First-impression click behavior is the strongest predictor of future fatigue

**Implications:** These findings enable data-driven frequency cap recommendations (3-5 exposures), personalized ad exposure strategies, and improved ROI through fatigue-aware optimization.

---

## Data Source and License

### Criteo Attribution Dataset

| Attribute | Details |
|-----------|---------|
| **Provider** | Criteo AI Lab |
| **Records** | 16,468,027 impressions |
| **Time Period** | 30 days |
| **Format** | TSV (gzipped) |

### Relevant Links

| Resource | URL |
|----------|-----|
| **Data Download** | [Criteo Attribution Modeling Dataset](http://ailab.criteo.com/criteo-attribution-modeling-bidding-dataset/) |
| **Criteo AI Lab** | [https://ailab.criteo.com/](https://ailab.criteo.com/) |
| **Dataset Documentation** | [README in data/raw/](data/raw/README.md) |
| **Criteo Terms of Service** | [https://www.criteo.com/terms-and-conditions/](https://www.criteo.com/terms-and-conditions/) |
| **Criteo Privacy Policy** | [https://www.criteo.com/privacy/](https://www.criteo.com/privacy/) |

### Data License

The Criteo Attribution Dataset is released under **Criteo's Research Data License**:

> *"This dataset is released for academic and research purposes only. It may not be used for commercial purposes without explicit permission from Criteo."*

**Key Terms:**
- Academic/research use permitted
- Publications using this data must cite Criteo
- Commercial use prohibited without permission
- Re-distribution of raw data prohibited

**Citation Required:**
```bibtex
@misc{criteo_attribution_dataset,
  author = {Criteo AI Lab},
  title = {Criteo Attribution Modeling Dataset},
  year = {2017},
  howpublished = {\url{http://ailab.criteo.com/criteo-attribution-modeling-bidding-dataset/}}
}
```

### Dataset Schema

| Column | Description | Type |
|--------|-------------|------|
| `timestamp` | Impression time (seconds from start) | Integer |
| `uid` | Hashed user identifier | Integer |
| `campaign` | Campaign identifier | Integer |
| `click` | Click label (0/1) | Binary |
| `conversion` | Conversion label (0/1) | Binary |
| `cat1-cat9` | Hashed categorical features | Integer |

Full schema documentation: [data/raw/README.md](data/raw/README.md)

---

## Project Structure

```
creative_fatigue_analysis/
├── data/
│   ├── raw/                    # Criteo Attribution Dataset (not included in repo)
│   │   └── README.md           # Dataset documentation
│   ├── processed/              # Feature-engineered data
│   └── samples/                # Fatigue-optimized samples
├── notebooks/
│   ├── 01_data_acquisition.ipynb      # Data loading & optimized sampling
│   ├── 02_exploratory_analysis.ipynb  # EDA and data quality
│   ├── 03_feature_engineering.ipynb   # Feature creation (uses src/)
│   ├── 04_rq1_ctr_fatigue_analysis.ipynb   # RQ1: CTR vs exposure
│   ├── 05_rq2_fatigue_resistance.ipynb     # RQ2: Fatigue by category
│   ├── 06_rq3_predicting_fatigue.ipynb     # RQ3: Predictive modeling
│   └── 07_final_report.ipynb          # Final report with all findings
├── src/
│   ├── data_loader.py          # Data loading utilities
│   ├── feature_engineering.py  # Feature computation functions
│   ├── smart_sampling.py       # Fatigue-optimized sampling
│   ├── models.py               # Baseline, time-aware, decay models
│   ├── evaluation.py           # Metrics and visualization
│   └── utils.py                # Helper functions
├── config/
│   └── config.yaml             # Configuration parameters
├── results/
│   ├── figures/                # All visualizations
│   └── tables/                 # Summary statistics (CSV)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## Installation

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd creative_fatigue_analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download the Dataset

1. Visit [Criteo Attribution Modeling Dataset](http://ailab.criteo.com/criteo-attribution-modeling-bidding-dataset/)
2. Download `criteo_attribution_dataset.tsv.gz`
3. Place in `data/raw/` directory

### 3. Run the Analysis

Execute notebooks in order:

| Notebook | Description | Input | Output |
|----------|-------------|-------|--------|
| `01_data_acquisition.ipynb` | Load data, create optimized sample | Raw TSV | `samples/criteo_fatigue_optimized.csv` |
| `02_exploratory_analysis.ipynb` | EDA, data quality checks | Sample CSV | `processed/data_with_exposures.csv` |
| `03_feature_engineering.ipynb` | Create 40+ features | Exposures CSV | `processed/data_with_all_features.csv` |
| `04_rq1_ctr_fatigue_analysis.ipynb` | Analyze CTR decline | Features CSV | `results/tables/rq1_*.csv` |
| `05_rq2_fatigue_resistance.ipynb` | Compare fatigue by category | Features CSV | `results/tables/rq2_*.csv` |
| `06_rq3_predicting_fatigue.ipynb` | Build predictive models | Features CSV | `results/tables/rq3_*.csv` |
| `07_final_report.ipynb` | Synthesize all findings | All results | Final report |

---

## Research Questions and Findings

### RQ1: How does CTR change with repeated exposure?

| Exposure | CTR | Decline from Baseline |
|----------|-----|----------------------|
| 1 (Baseline) | 52.9% | — |
| 2 | 44.8% | -15.3% |
| 3 | 41.6% | -21.3% |
| 5 | 38.5% | -27.1% |

**Finding:** CTR declines 25-27% by the 5th exposure using cohort-based within-user analysis.

### RQ2: Are some ads more fatigue-resistant?

| Campaign Type | CTR Decline | Resistance |
|--------------|-------------|------------|
| High CTR | 22.8% | Most Resistant |
| Medium CTR | 30.2% | Moderate |
| Low CTR | 35.6% | Least Resistant |

**Finding:** High-performing campaigns show 12.8 percentage points less fatigue.

### RQ3: Can we predict fatigue?

| Model | AUC-ROC | Accuracy |
|-------|---------|----------|
| Logistic Regression | 0.707 | 76.1% |
| XGBoost | 0.703 | 76.7% |
| Random Forest | 0.703 | 76.5% |

**Finding:** First-click behavior accounts for 66% of predictive power.

---

## Key Dependencies

```
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
xgboost>=2.0.0 (optional)
```

See `requirements.txt` for complete list.

---

## References

1. **Criteo Attribution Dataset**  
   Criteo AI Lab. (2017). *Criteo Attribution Modeling Dataset*.  
   http://ailab.criteo.com/criteo-attribution-modeling-bidding-dataset/

2. **Pechmann, C., & Stewart, D. W. (1988)**  
   Advertising repetition: A critical review of wearin and wearout.  
   *Current Issues and Research in Advertising*, 11(1-2), 285-329.

3. **Schmidt, S., & Eisend, M. (2015)**  
   Advertising repetition: A meta-analysis on effective frequency in advertising.  
   *Journal of Advertising*, 44(4), 415-428.

4. **Sahni, N. S., Narayanan, S., & Kalyanam, K. (2019)**  
   An experimental investigation of the effects of retargeted advertising.  
   *Journal of Marketing Research*, 56(3), 401-418.

---

## License

### Code License

This project's code is released under the **MIT License**:

```
MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

### Data License

The Criteo Attribution Dataset is subject to Criteo's Research Data License (see [Data License](#data-license) section above).

---

## Contact

For questions about this analysis, please open an issue in this repository.

For questions about the Criteo dataset, contact: [ailab@criteo.com](mailto:ailab@criteo.com)
