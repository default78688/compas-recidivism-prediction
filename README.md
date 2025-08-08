#  Recidivism Prediction Using Machine Learning

This project focuses on building machine learning models to predict the likelihood of recidivism — whether a person will reoffend within two years. We use the **COMPAS** dataset, which includes demographic information, criminal history, and charge-related data for individuals processed by the justice system.

The motivation behind this project is to explore whether ML models can assist in understanding recidivism patterns, while also shedding light on the fairness and bias issues that arise in criminal justice applications.

---

##  Dataset

The dataset used is the **COMPAS (Correctional Offender Management Profiling for Alternative Sanctions)** dataset, publicly available and frequently used in fairness studies.

### Features used:
- Demographics: `sex`, `age`, `age_cat`, `race`
- Criminal history: `priors_count`, `juv_fel_count`, `juv_misd_count`, `juv_other_count`
- Charge details: `c_charge_degree`
- Engineered features: `arrest_month`, `arrest_season`

### Target:
- `two_year_recid` (1 = reoffended within 2 years, 0 = did not)

---

##  Project Structure

1. **Data Cleaning & Feature Engineering**
   - Handled missing values
   - Extracted date-based features (month, season of arrest)
   - One-hot encoding for categorical variables

2. **Exploratory Data Analysis (EDA)**
   - Visualized distributions of age and prior counts
   - Analyzed recidivism rates by race, age category, and charge degree
   - Highlighted data imbalances and possible social biases

3. **Model Training & Evaluation**
   - Trained several classifiers:
     - Logistic Regression
     - Random Forest
     - Gradient Boosting
     - XGBoost
     - K-Nearest Neighbors
   - Standardized numerical inputs
   - Evaluated using:
     - **Accuracy**
     - **AUC (Area Under Curve)**

4. **Results**
   - Gradient Boosting showed best performance:  
     - Accuracy ≈ 69%  
     - AUC ≈ 74%
   - Neural network was also tested and showed similar performance.

---

##  Visualizations

-  **Age Distribution**: Most individuals are aged between 20-40.
-  **Prior Offenses**: Most have low prior counts, but some have high.
-  **Recidivism by Race**: Used to investigate fairness across racial groups.
-  **Recidivism by Age Category**: Younger individuals tend to reoffend more.
-  **Charge Degree vs Recidivism**: Severity of charge vs recidivism probability.

---

##  Key Insights

- Feature engineering and proper preprocessing are essential for performance.
- Models like Gradient Boosting and XGBoost outperform simpler models like Logistic Regression.
- There’s a clear potential for bias and unfair outcomes, especially when race is included as a feature — this needs careful ethical consideration.

---

##  Conclusion

This project demonstrates that it’s possible to predict recidivism with moderate accuracy using basic features. However, **fairness, transparency, and accountability** must be central to any real-world application.

This notebook is a **starting point** for deeper studies in:
- Fairness in machine learning
- Bias mitigation techniques
- Ethical AI systems in justice

---

##  Future Work

- Hyperparameter optimization
- More advanced neural networks or ensemble stacking
- Feature selection techniques
- Bias mitigation algorithms
- Legal and ethical audit of model decisions

---

##  Contact

If you'd like to connect or discuss this project, feel free to reach out:
- Contact me via Email: karimiabolfazl466@gmail.com  
- Telegram: [@Abolfazlk83](https://t.me/Abolfazlk83)  
- LinkedIn: Coming soon  
- GitHub: [github.com/abolfazlkarimi83](https://github.com/abolfazlkarimi83)

---

## 🙏 Tanks

Thanks for reviewing this project.  
This notebook was developed by **Abolfazl Karimi** as part of a self-study journey in machine learning and time series forecasting.
