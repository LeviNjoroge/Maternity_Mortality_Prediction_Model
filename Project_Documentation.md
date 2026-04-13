# Maternal Mortality Prediction Project

## Overview
This machine learning project aims to analyze and forecast maternal mortality trends globally using dataset records spanning from 1990 to 2021. By leveraging demographic profiles, region categorization, and historical health event data, the system builds predictive models to foresee the expected **Maternal Mortality Ratios (deaths per 100,000 live births)**. 

The project pipeline has been unified into a single extensive workflow file (`Maternity_Mortality_Prediction_Model.ipynb`) built out of four distinct machine learning modeling approaches, each functioning as an autonomous module spanning **Data Preparations**, **Model Building**, **Model Evaluation**, **Feature Analysis**, and **Data Visualization**.

---

## Dataset Description
**Source File:** `Maternal_Mortality.csv`
- **Total Records:** 195 Country/Region profiles
- **Features:** 39+ Features containing geo-demographic statistics and multi-decade time-series data.
- **Key Columns Included:**
  - `ISO3`, `Country`, `Continent`, `Hemisphere`
  - `Human Development Groups`, `UNDP Developing Regions`, `HDI Rank (2021)`
  - Annual historical targets: `Maternal Mortality Ratio (1990)` through `Maternal Mortality Ratio (2021)`

---

## Machine Learning Algorithms Used

The core of the project relies on four dedicated algorithms specifically selected for distinct capabilities in handling tabular time-series health metrics. 

### 1. Linear Regression (Module 1)
- **Description:** A primary statistical modeling approach to establish the baseline causal trend between past years of mortality inputs and final outcome mappings.
- **Goal:** Learns linear weight assignments across historical years (e.g., how heavily does 2018 or 2019 correlate mathematically to the 2021 outcome).
- **Evaluation Metrics:** Analyzed via R-squared score, Residual interpretations, and coefficient weights to determine directly proportional year-over-year impact.

### 2. Random Forest Regressor & Risk Classifier (Module 2)
- **Description:** An ensemble learning method generating multiple decision trees to mitigate overfitting phenomena commonly seen in smaller country datasets. 
- **Goal:** 
  - **Risk Classification:** Converts patient/country risk into three distinct categories (`Low Risk`, `Mid Risk`, `High Risk`) utilizing factors like health expenditures, HDI distributions, or synthetic patient demographics. 
  - **Future Horizon Regressor:** Configured to predict distinct temporal horizons ($t+1, t+3, t+5, t+10$ years) while outputting highly reliable 90% confidence intervals derived from intra-tree variance.

### 3. XGBoost (Module 3)
- **Description:** eXtreme Gradient Boosting optimizes sequential tree correction to maximize forecasting accuracy, especially resilient to incomplete records.
- **Data Engineering:** Engineers rolling analytical variables natively into its training: `lag_1`, `lag_2`, `lag_3`, `rolling_mean_5`, and `mmr_change`.
- **Evaluation Metrics:** Validation performed comparing Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and standard $R^2$ scores to highlight improvements over standard Random Forest approaches over non-linear data distributions.

### 4. ARIMA - Time Series Forecasting (Module 4)
- **Description:** AutoRegressive Integrated Moving Average (`statsmodels.tsa.arima.model.ARIMA`). Designed inherently for time-bound datasets that exhibit distinct autoregressive trends.
- **Goal:** Targets localized execution (e.g., specifically targeting a country like `Kenya`) explicitly separating its temporal index values (`kenya_ts`) and projecting moving forecasts (steps=5 into 2026).
- **Why it’s unique here:** Unlike XGBoost and RF which require engineered discrete "lag" columns for pseudo-series forecasting, ARIMA statistically evaluates raw continuous historical strings to estimate purely temporal drifts.

---

## Unified Module Architecture

Each of the four models above strictly follows the standardized architectural schema required for robust replication:

1. **Data Preparations:**
   - Filtering the 1990-2021 column dimensions via string manipulation.
   - Transposing horizontal country records into vertical time series objects depending on the model (e.g., ARIMA requires a strict column drop).
   - Establishing rolling statistical calculations and handling `NaN` historical lapses.
2. **Model Building:**
   - Train/Test stratification maintaining temporal continuity.
   - Initializing estimators holding optimal hyperparameters.
3. **Model Evaluation:**
   - Generating definitive accuracy footprints (MAE, MSE, $R^2$, or strict class probabilities).
4. **Feature Analysis:**
   - Plotting feature importances (e.g., identifying whether the generic `HDI Rank` or `lag_1` variable drove the outcome the most).
5. **Data Visualization:**
   - Rendering comparative distribution plots, Matplotlib temporal trajectory lines (e.g., 1990 vs 2026 forecast extrapolations) and confidence threshold mappings.

---

## Repository Structure

```text
📦 GroupAssignment
 ┣ 📜 1.ipynb
 ┣ 📜 2.ipynb
 ┣ 📜 3.ipynb
 ┣ 📜 4.ipynb
 ┣ 📜 Maternal_Mortality.csv
 ┣ 📜 Maternity_Mortality_Prediction_Model.ipynb
 ┣ 📜 Maternity Mortality Prediction Model.docx
 ┣ 📜 SOEN398-Group-Project.pdf
 ┣ 📜 README.md
 ┗ 📜 Project_Documentation.md
```

### File Descriptions

- **`1.ipynb`**: Contains the **Linear Regression** modeling module. Analyzes target distribution, feature correlation, and implements foundational predictive modeling to establish a statistical baseline.
- **`2.ipynb`**: Contains the **Random Forest** (Regressor and Classifier) module. Defines the core patient prediction logic and multi-horizon time-series regressors, evaluating confidence intervals derived from decision tree variance.
- **`3.ipynb`**: Contains the **XGBoost** modeling module. Leverages engineered temporal features (e.g., rolling means, lag indicators) to predict non-linear outcomes and handles missing metrics inherently.
- **`4.ipynb`**: Contains the **ARIMA** modeling module. A pure time-series approach isolating individual country-level arrays (e.g., Kenya mortality data) for temporal $t+5$ forecasting into 2026.
- **`Maternal_Mortality.csv`**: The primary dataset powering all models. Comprised of 71 columns of demographic and health features for 195 countries spanning the years 1990 to 2021.
- **`Maternity_Mortality_Prediction_Model.ipynb`**: The **Final Unified Notebook**. All four modules (1–4) have been mechanically merged sequentially into this master file under standardized subheadings (*Data Preparations, Model Building, Evaluation, Feature Analysis, Data Visualization*). This is the main deliverable.
- **`Project_Documentation.md`**: This current file, providing a detailed high-level summary of the overall architecture, algorithmic approaches, the dataset, and the workspace layout.
- **`Maternity Mortality Prediction Model.docx`**: The written group report and analytical write-up discussing the methodology, results, and implementations within the code.
- **`SOEN398-Group-Project.pdf`**: The course assignment rubric and syllabus instructions detailing the constraints and requirements for this final AI/ML group project.
- **`README.md`**: General repository markdown containing setup instructions or high-level introductory notes.