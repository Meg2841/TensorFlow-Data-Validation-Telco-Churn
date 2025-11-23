# TensorFlow Data Validation – Telco Customer Churn

This project is an extended version of a TensorFlow Data Validation (TFDV) lab.  
The original lab demonstrates how to use TFDV on a dataset.  
This version adds a second case study using the Telco Customer Churn dataset and a few extra utilities.

The goal is to show how TFDV can be used to explore, validate, and slice real-world data in a way that is closer to production workflows.

---

## Repository structure

```
.
├─ TFDV_Lab1.ipynb           # Main notebook: original lab + Telco extension
├─ util.py                   # Helper functions from the original lab
├─ report_utils.py           # New helper functions for feature/slice summaries
├─ Telco-Customer-Churn.csv  # Telco dataset used in the extension
├─ data/                     # Original lab data (unchanged)
├─ img/                      # Images used by the original lab
└─ ...                       # Jupyter checkpoints and other support files
```

---

## Datasets

### 1. Original lab dataset

The original TFDV lab uses a small tabular dataset stored under `data/`.  
It is kept as-is to preserve the original exercise and to contrast with the Telco example.

The original notebook walks through:

- Computing statistics with `tfdv.generate_statistics_from_csv`
- Inferring a schema with `tfdv.infer_schema`
- Validating evaluation data against that schema
- Performing simple slicing and anomaly inspection

### 2. Telco Customer Churn dataset

The extended part of the lab uses the Telco Customer Churn dataset.  
Source: IBM / Kaggle Telco Customer Churn dataset.  
The CSV is included in this repository as:

```
Telco-Customer-Churn.csv
```

Each row represents a customer of a telecom provider.  
Columns include:

- `customerID`: unique customer identifier  
- `gender`, `SeniorCitizen`, `Partner`, `Dependents`  
- `tenure`: number of months the customer has stayed with the company  
- `PhoneService`, `InternetService`, `Contract`, `PaymentMethod`  
- `MonthlyCharges`, `TotalCharges`  
- `Churn`: whether the customer left (Yes/No)

This dataset is used to demonstrate how TFDV behaves on a more realistic, mixed-type dataset with both numeric and categorical features.

---

## What the notebook does

### Original lab section

The first part of `TFDV_Lab1.ipynb` follows the provided lab:

- Loads the original dataset from `data/`
- Generates statistics using TFDV
- Infers a schema and displays it
- Validates evaluation data and shows anomalies
- Demonstrates basic slicing and anomaly exploration


### Telco extension section

The second part of the notebook adds a new workflow built around the Telco dataset.  
The key steps are:

1. **Load and clean Telco data**

   - The file `Telco-Customer-Churn.csv` is loaded into a pandas DataFrame.
   - The `TotalCharges` column is converted to numeric.
   - Invalid/blank `TotalCharges` values are coerced to `NaN` and then filled with zero so that the column can be treated as numeric.

2. **Generate statistics using the DataFrame API**

   - Instead of reading from a CSV, statistics are generated with  
     `tfdv.generate_statistics_from_dataframe(df_telco)`.
   - This shows how TFDV integrates directly with pandas workflows.

3. **Infer and inspect a new schema**

   - A new schema is inferred from the Telco statistics.
   - The schema describes each feature’s type and domain, which may differ significantly from the original lab dataset.
   - The schema is displayed to understand how TFDV interprets the Telco data.

4. **Create tenure-based slices**

   - The dataset is split into two groups:
     - Short-tenure customers: `tenure < 12`
     - Long-tenure customers: `tenure >= 12`
   - This allows analysis of how feature distributions differ between customer groups.

5. **Generate and compare slice statistics**

   - TFDV generates statistics for each slice separately.
   - A comparison view visualizes distributions for short-tenure and long-tenure customers side by side.
   - This helps reveal differences in features such as charges and churn rates across tenure groups.

6. **Add custom numeric constraints to the schema**

   - For selected numeric features such as `MonthlyCharges` and `TotalCharges`, custom `min` and `max` ranges are added to the inferred schema.
   - These constraints make the schema stricter and allow validation to flag values that fall outside expected numeric ranges.

7. **Validate Telco statistics against the stricter schema**

   - TFDV validates the Telco statistics using the modified schema.
   - The anomaly view shows whether any features violate the new constraints or exhibit unexpected behavior.
   - In the current configuration, the dataset passes the stricter checks, which is also a useful result.

8. **Slice-level drift-style comparison **

   - Short-tenure statistics are treated as a reference, and long-tenure statistics as an evaluation set.
   - TFDV’s visualization is used to compare distributions between these slices.
   - Validation against the Telco schema for the evaluation slice can highlight any drift-like differences between customer groups.

9. **Additional utilities via `report_utils.py`**

   - A new Python module `report_utils.py` is added.
   - It contains small helper functions to build compact summaries:
     - `summarize_feature(df, feature)` produces basic stats (count, unique, missing, and numeric range when applicable).
     - `compare_slices_mean(df1, df2, feature, ...)` compares the mean of a numeric feature between two slices.
   - The notebook uses these helpers to:
     - Create a summary table for important Telco features such as `tenure`, `MonthlyCharges`, `TotalCharges`, and `Churn`.
     - Compare average values of numeric features between short-tenure and long-tenure customers.
   - These summaries complement the TFDV visualizations with simple numerical views.

---

## How to run the notebook locally

TFDV currently works best with specific versions of Python, TensorFlow, and protobuf.  
Running the notebook in Google Colab is not recommended for this project because of version incompatibilities.

The steps below describe how to run everything locally using Anaconda and Python 3.9, which is known to work with this setup.

### 1. Clone the repository

```bash
git clone https://github.com/Meg2841/TensorFlow-Data-Validation-Telco-Churn.git
cd TensorFlow-Data-Validation-Telco-Churn
```

### 2. Create and activate a conda environment (Python 3.9)

```bash
conda create -n tfdv_env python=3.9 -y
conda activate tfdv_env
```

### 3. Install required packages

The following combination of packages has been tested with this notebook:

```bash
pip install   pandas   scikit-learn   jupyterlab   "protobuf==3.20.3"   "tensorflow==2.12.0"   "apache-beam==2.48.0"   "tensorflow-data-validation==1.13.0"
```

If you encounter protobuf-related errors, you may also need to set:

```bash
set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python    # Windows (cmd)

```

before starting Jupyter.

### 4. Launch JupyterLab

From inside the environment and repo folder:

```bash
jupyter lab
```

In the JupyterLab interface, open:

```text
TFDV_Lab1.ipynb
```

and run the cells from top to bottom.

The Telco extension section should both run and display all TFDV outputs, including:

- Statistics visualizations  
- Inferred schema views  
- Anomaly reports (if any)  
- Slice comparison plots  
- Summary tables generated by `report_utils.py`

---

## How to interpret the results

- Use the statistics and schema views to understand the structure of each dataset.
- Look at the Telco slice comparison to see how tenure affects distributions, charges, and churn.
- Inspect the schema constraints and validation results to understand how TFDV can enforce basic data quality rules.
- Use the summary tables from `report_utils.py` as quick reference points for reporting and discussion.

---

---

## License and data usage

The Telco Customer Churn dataset is provided by IBM and distributed on Kaggle.  
Please review the original dataset license and usage terms on the dataset’s source page if you plan to reuse it.

This repository is intended for educational and experimentation purposes, demonstrating how to apply TensorFlow Data Validation to a real-world dataset.
