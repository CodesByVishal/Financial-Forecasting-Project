# Financial Forecasting Frontier — Distributed Machine Learning with PySpark
---

## Project Overview

Demonstrates Apache PySpark for banking analytics at scale — processing 45,211 client records through a 4-task distributed ML pipeline. The project shows how Spark enables batch analytics, distributed model training, and real-time streaming predictions that are impractical on single-machine tools.

---

## Dataset — bank.csv

Portuguese Bank Marketing Campaign dataset:

| Property | Detail |
|---|---|
| Records | 45,211 client records |
| Features | 17 (demographics + campaign + economic context) |
| Target | `y` — binary (yes/no term deposit subscription) |
| Class balance | ~88% No / ~12% Yes |
| Missing values | None |

Key features: age, job, marital status, education, balance, contact method, campaign count, CPI, unemployment rate, euribor3m.

---

## 4-Task Pipeline

### Task 1 — Efficient Data Parallelism

- `spark.sql.shuffle.partitions = 4`
- `df.repartition(4, 'balance')` — balance column chosen for even distribution
- Parallel `groupBy().agg()` across 4 partitions simultaneously
- **Result:** Management category = highest average balance; Age 40–50 = highest total loan amounts
- Spark Event Logging configured for resource monitoring

### Task 2 — EDA with Spark SQL

- `createOrReplaceTempView('bank')` → Spark SQL queries
- UDF for age group categorisation
- **Findings:**
  - University-educated clients: highest subscription rate
  - Cellular contact outperforms telephone contact
  - May = highest outreach volume; June/October = better conversion rates
  - Married clients: highest absolute subscription count

### Task 3 — Spark ML Pipeline (3 Models Compared)

Pipeline stages:
1. `StringIndexer` — categorical columns to numeric indices
2. `OneHotEncoder` — indices to binary vectors (no false ordinal relationships)
3. `VectorAssembler` — all features into single `features` vector
4. Classifier (pluggable — swap model without changing preprocessing)

| Model | Strength | Result |
|---|---|---|
| Logistic Regression | Interpretable coefficients | Moderate accuracy |
| **Random Forest (100 trees)** | **Non-linear + feature importance** | **Best AUC** |
| GBT Classifier | Competitive accuracy | Slightly slower inference |

**Hyperparameter Tuning:** `ParamGridBuilder` + `CrossValidator(numFolds=5)` — distributed grid search across the cluster. Optimal: `numTrees=100`, `maxDepth=5`.

### Task 4 — Real-Time Streaming with Structured Streaming

- `spark.readStream` on bank.csv chunks (simulates live transaction feed)
- Pre-trained RF model broadcast to all executors via `SparkContext.broadcast()`
- 10-minute sliding window: rolling average balance and transaction count
- **Watermarking** (10 minutes) handles late/out-of-order data
- Predictions written to console output stream

---

## Key Results

| Task | Key Finding |
|---|---|
| Task 1 | Management = highest avg balance; Age 40–50 = highest loan exposure |
| Task 2 | University education → highest subscription rate |
| Task 3 | Random Forest best AUC; optimal: numTrees=100, maxDepth=5 |
| Task 4 | RF broadcast to executors; 10-min rolling window stable |

---

## Tech Stack

```
Python | PySpark | Spark ML | Spark SQL | Spark Structured Streaming
Pandas | Matplotlib | Seaborn | Scikit-learn (local evaluation)
```

---

## Repository Contents

```
Financial-Forecasting-Project/
├── Financial_Forecasting_Frontier_Distributed_ML.ipynb   # Main PySpark notebook
├── bank.csv                                               # Bank marketing dataset
├── Data Analysis and Management using Hadoop & Hive.pdf  # Supporting reference
├── Financial_Forecasting_Presentation.pptx               # Presentation (12 slides)
└── README.md
```

---

## How to Run

```bash
pip install pyspark pandas matplotlib seaborn
jupyter notebook Financial_Forecasting_Frontier_Distributed_ML.ipynb
```

A local Spark session is initialised within the notebook — no external cluster required for local execution.

---

*AlmaBetter M.Sc. Data Science | 2026 | Vishal Kumar Singh*
