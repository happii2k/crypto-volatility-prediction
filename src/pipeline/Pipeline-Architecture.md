# Pipeline Architecture Document
## Cryptocurrency Volatility Prediction Project

### Document Information
- **Project**: Cryptocurrency Volatility Prediction
- **Version**: 1.0
- **Date**: August 13, 2025
- **Author**: ML Engineering Team

---

## 1. Overview

This document details the end-to-end pipeline architecture for the Cryptocurrency Volatility Prediction project. It covers every stage from raw data ingestion to real-time prediction delivery in the Streamlit application. The pipeline is designed to be modular, reproducible, and easy to maintain.

---

## 2. High-Level Pipeline Diagram
```
┌────────────────────────────────────────────────────────────────────────────┐
│                             OFFLINE WORKFLOW                               │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌──────────┐   ┌────────────┐   ┌──────────────┐   ┌───────────────┐      │
│  │ Raw CSV  │ → │ Data        │ → │ Feature       │ → │ Model Training │      │
│  │  Files   │   │ Processing  │   │ Engineering   │   │  & Selection  │      │
│  └──────────┘   └────────────┘   └──────────────┘   └───────────────┘      │
│        │              │                │                    │             │
│        ▼              ▼                ▼                    ▼             │
│  ┌──────────┐   ┌────────────┐   ┌──────────────┐   ┌─────────────────┐   │
│  │ Cleaned  │   │ Processed   │   │ Feature       │   │ crypto_volatility │   │
│  │  Data    │   │ DataFrame   │   │ Matrix (X)    │   │ _model.pkl       │   │
│  └──────────┘   └────────────┘   └──────────────┘   └─────────────────┘   │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
                                       ▲
                                       │ Model Artifact
                                       │
┌────────────────────────────────────────────────────────────────────────────┐
│                              ONLINE WORKFLOW                               │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Streamlit UI → Input Validation → Feature Engineering → Model Inference → │
│  Visualization & Metrics                                                   │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Stage-by-Stage Breakdown

### 3.1 Data Ingestion
| Step | Component | Description |
|------|-----------|-------------|
| 1 | **Raw CSV Import** | Historical OHLC, volume, and market-cap data are ingested from `dataset.csv` or similar files. |
| 2 | **Schema Validation** | Pandas checks ensure required columns exist (`date`, `open`, `high`, `low`, `close`, `volume`, `marketCap`, `crypto_name`). |

### 3.2 Data Processing
| Step | Operations |
|------|------------|
| Missing-Value Handling | SimpleImputer (median strategy) fills gaps in numeric columns. |
| Outlier Treatment | IQR capping for price, volume, and market-cap related features. |
| Type Conversion | `date` → `datetime64`; categorical encoding for `crypto_name`. |

### 3.3 Feature Engineering
| Feature Category | Details |
|------------------|---------|
| Price Dynamics | `price_change`, `range`, `return_pct` |
| Market Dynamics | `volume_change`, `marketCap_change` |
| Volatility Indicators | 7-, 14-, 30-day rolling std; Garman-Klass estimator |
| Temporal Features | `day_of_week`, `month`, `quarter`, `date_ordinal` |
| Encodings | Label-encoded `crypto_encoded` |

### 3.4 Feature Selection
1. **Correlation Filter**: Drops features with absolute correlation > 0.9.
2. **Target Leakage Guard**: Removes `volatility_30d` (future information).
3. **Redundancy Reduction**: Eliminates duplicate temporal or frequency-based columns.

### 3.5 Model Training & Selection
| Phase | Action |
|-------|--------|
| Train/Test Split | 80 / 20 shuffle split with fixed seed. |
| Imputation & Scaling | Median imputation → StandardScaler (linear models). |
| Algorithms | Linear, Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting, XGBoost, SVR. |
| Evaluation | `R²`, `RMSE`, `MAE` on test set. |
| Selection | Highest R² model (XGBoost by default). |
| Persistence | `joblib.dump` saves model, scaler, imputer, label encoder, feature list to **crypto_volatility_model.pkl**. |

### 3.6 Deployment Workflow
| Component | Description |
|-----------|-------------|
| **Streamlit Front-End** | Provides inputs for crypto, date, OHLC, volume, market-cap. |
| **Input Validator** | Ensures logical price bounds & positive values. |
| **On-the-fly Feature Engineering** | Mirrors offline logic to build the same feature vector. |
| **Model Loader** | Loads `.pkl` once per session (cached). |
| **Prediction Engine** | Applies imputer → model.predict. |
| **Visualization Layer** | Plotly gauge + historical volatility line. |

---

## 4. Data Storage Artifacts
| File | Purpose |
|------|---------|
| `dataset.csv` | Primary historical market data source. |
| `crypto_volatility_model.pkl` | Serialized model and preprocessing objects. |
| `crypto_volatility_trainer.py` | Offline training script (re-train anytime). |
| `streamlit_app.py` | Online prediction service. |

---

## 5. Operational Flow
1. **Retrain Cycle** (e.g., monthly):
   ```bash
   python crypto_volatility_trainer.py  # regenerates .pkl
   git commit -m "Retrained with latest data"
   ```
2. **Deployment Cycle**:
   ```bash
   streamlit run streamlit_app.py       # local or cloud
   ```
3. **Prediction Request**: User fills sidebar → backend builds feature vector → model inference → result visualization (< 2 s latency).

---

## 6. Monitoring & Maintenance
- **Model Drift Checks**: Compare live R² against offline benchmark; trigger retrain if drop > 10%.
- **Logging**: `crypto_volatility.log` captures prediction requests & errors.
- **Alerts**: Optional integration with Slack/Webhooks on critical failures.

---

## 7. Conclusion
This pipeline ensures a repeatable path from raw market data to end-user volatility forecasts, satisfying all project deliverables: data processing, feature engineering, model training, evaluation, and compulsory deployment via Streamlit.
