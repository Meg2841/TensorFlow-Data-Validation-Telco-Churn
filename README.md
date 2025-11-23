# TensorFlow-Data-Validation-Telco-Churn

# TensorFlow Data Validation – Telco Customer Churn

This project is an extended version of a TensorFlow Data Validation (TFDV) lab.  
The original lab demonstrates how to use TFDV on a dataset.  
This version adds a second case study using the Telco Customer Churn dataset and a few extra utilities.

The goal is to show how TFDV can be used to explore, validate, and slice real-world data in a way that is closer to production workflows.

---

## Repository structure

```text
.
├─ TFDV_Lab1.ipynb           # Main notebook: original lab + Telco extension
├─ util.py                   # Helper functions from the original lab
├─ report_utils.py           # New helper functions for feature/slice summaries
├─ Telco-Customer-Churn.csv  # Telco dataset used in the extension
├─ data/                     # Original lab data (unchanged)
├─ img/                      # Images used by the original lab
└─ ...                       # Jupyter checkpoints and other support files

Datasets
1. Original lab dataset

The original TFDV lab uses a small tabular dataset stored under data/.
It is kept as-is to preserve the original exercise and to contrast with the Telco example.
The original notebook walks through:
- Computing statistics with tfdv.generate_statistics_from_csv
- Inferring a schema with tfdv.infer_schema
- Validating evaluation data against that schema
- Performing simple slicing and anomaly inspection

2. Telco Customer Churn dataset (new)

The extended part of the lab uses the Telco Customer Churn dataset.
Source: IBM / Kaggle Telco Customer Churn dataset.
The CSV is included in this repository as:
