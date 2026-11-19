# 📦 SmartStock – Smart Inventory and Demand Prediction System

### An End-to-End Machine Learning Project for Retail Demand Forecasting and Inventory Decision Support

SmartStock is a machine learning-based system that forecasts retail product demand for the next 7 and 30 days and provides inventory recommendations to help retailers make better stocking decisions.

The system uses historical sales patterns, promotional information, time-based features, and machine learning to estimate future demand and identify products that may require replenishment. An interactive Streamlit dashboard presents forecasts, inventory insights, and model performance.

---

## 🚀 Project Demo

<!-- Add your Streamlit dashboard screenshot here -->

![SmartStock Dashboard](images/dashboard.png)

*Replace this image with a screenshot of your actual Streamlit dashboard.*

**Application:** Runs locally using Streamlit.

**GitHub Repository:** Add your repository URL here.

---

## 🎯 Problem Statement

Retail businesses must maintain sufficient inventory to meet customer demand while avoiding overstocking and unnecessary storage costs.

However, demand varies across products, stores, seasons, and promotional periods. Poor demand estimation can lead to stockouts, excess inventory, and inefficient purchasing decisions.

SmartStock addresses this problem by using machine learning to forecast product demand and generate inventory replenishment recommendations.

**Target users:**
- Retail store managers
- Inventory planners
- Retail operations teams
- Small and medium-sized retailers

**Project objective:** Provide data-driven demand forecasts and inventory recommendations to support better retail inventory planning.

---

## 📊 Dataset

**Dataset:** Store Sales – Time Series Forecasting

**Source:** [Kaggle – Store Sales Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)

The original dataset contains historical sales information from Favorita grocery stores in Ecuador.

### Dataset Details

| Attribute | Description |
|---|---|
| Original file | `train.csv` |
| Original records | 3,000,888 |
| Number of stores | 54 |
| Product families | 33 |
| Original date range | January 2013 – August 2017 |
| Target variable | `sales` |
| Problem type | Time-series regression |

### Important Features

| Feature | Description |
|---|---|
| `date` | Sales transaction date |
| `store_nbr` | Store identifier |
| `family` | Product category |
| `sales` | Total product-family sales |
| `onpromotion` | Number of items on promotion |

### Data Extension

To experiment with more recent forecasting periods, synthetic records were generated for 2018–2026 using historical sales patterns.

The combined dataset contains approximately **8.75 million records**.

**Important:** The original Kaggle data ends in August 2017. Records generated for 2018–2026 are synthetic, not actual observed retail sales. Therefore, model evaluation on the extended period is experimental and should not be interpreted as verified real-world performance.

---

## ⚙️ Project Workflow

The project follows an end-to-end machine learning workflow.

1. **Data Collection:** Obtained the original retail sales dataset from Kaggle.
2. **Exploratory Data Analysis:** Examined sales distributions, missing values, duplicate records, product categories, stores, and time-based trends.
3. **Data Preprocessing:** Converted date columns, organized the data, and prepared features for forecasting.
4. **Feature Engineering:** Created time-based and historical sales features.
5. **Model Development:** Trained and compared multiple regression models.
6. **Model Evaluation:** Used chronological validation and cross-validation to compare model performance.
7. **Hyperparameter Tuning:** Optimized the selected model.
8. **Demand Forecasting:** Generated recursive 7-day and 30-day demand forecasts.
9. **Inventory Analysis:** Calculated reorder points, safety stock, target stock, and recommended order quantities.
10. **Dashboard Development:** Built an interactive Streamlit application to present the results.

---

## 🧠 Feature Engineering

The following features were created to capture historical demand patterns:

| Feature | Purpose |
|---|---|
| `year` | Captures yearly trends |
| `month` | Captures monthly patterns |
| `day` | Represents the day of the month |
| `day_of_week` | Captures weekly demand patterns |
| `lag_1` | Previous day's sales |
| `lag_7` | Sales from seven days earlier |
| `lag_14` | Sales from fourteen days earlier |
| `rolling_mean_7` | Average sales over the previous seven days |

The lag and rolling features were calculated separately for each store and product family. The rolling mean uses previous observations to avoid incorporating the current target value.

---

## 🤖 Model Development and Comparison

Four approaches were evaluated:

- 7-Day Moving Average Baseline
- Decision Tree Regressor
- Random Forest Regressor
- HistGradientBoosting Regressor

The models were evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² score.

### Initial Model Comparison

The following results were obtained on the 2025 validation period.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| 7-Day Moving Average | 148.06 | 532.98 | 0.7609 |
| Decision Tree | 117.44 | 453.59 | 0.8268 |
| Random Forest | 106.68 | 405.05 | 0.8619 |
| HistGradientBoosting | 107.27 | 385.01 | 0.8752 |

### Cross-Validation Results

Expanding chronological cross-validation was used to examine model stability across multiple time periods.

| Model | Mean MAE | Mean RMSE | Mean R² |
|---|---:|---:|---:|
| Decision Tree | 120.17 | 463.72 | 0.8243 |
| Random Forest | 107.86 | 417.03 | 0.8579 |
| HistGradientBoosting | 108.00 | 395.73 | 0.8720 |

HistGradientBoosting was selected for further tuning because it provided a strong balance of prediction accuracy, stability, and training efficiency.

### Tuned Model Performance

After hyperparameter tuning, the HistGradientBoosting model achieved:

| Metric | Result |
|---|---:|
| MAE | 105.37 |
| RMSE | 383.10 |
| R² | 0.8765 |

The final model was trained using historical data up to 2025 and evaluated on the synthetic 2026 test period.

| Metric | Final Test Result |
|---|---:|
| MAE | 101.87 |
| RMSE | 367.28 |
| R² | 0.8814 |

**Evaluation note:** The 2026 test results are based on synthetic data. This evaluation uses one-step-ahead predictions with previous observations available for lag features; it is not the same as evaluating a fully recursive future forecast.

---

## 🔍 Key Insights and Findings

- The seven-day rolling average was the most influential feature in the permutation importance analysis.
- Promotional activity was another important factor in predicting sales.
- Historical sales from seven and fourteen days earlier contributed useful information.
- HistGradientBoosting achieved the strongest overall balance across the cross-validation metrics.
- Demand patterns differ across stores and product families, making store-level and category-level forecasting useful.
- Inventory recommendations can help identify product categories that may need replenishment.

### Feature Importance

Permutation importance was used to understand how individual features influence model performance.

The most influential features were:

1. `rolling_mean_7`
2. `onpromotion`
3. `lag_14`
4. `lag_7`
5. `lag_1`

Feature importance indicates predictive usefulness, not a causal relationship.

---

## 📦 Inventory and Reorder Recommendation System

SmartStock includes a rule-based inventory recommendation component that uses estimated demand to calculate replenishment quantities.

### Inventory Calculations

- **Average Daily Demand:** Estimated from historical sales.
- **Safety Stock:** Two days of average demand.
- **Lead Time:** Three days.
- **Reorder Point:** Lead-time demand plus safety stock.
- **Target Stock:** Seven days of average demand plus safety stock.
- **Recommended Order Quantity:** Difference between target stock and current stock, when positive.

### Reorder Logic

A product is assigned the status:

- **REORDER NOW:** Current stock is less than or equal to the reorder point.
- **SUFFICIENT STOCK:** Current stock is above the reorder point.

The current stock values are simulated because the original Kaggle dataset does not contain actual inventory records. Therefore, the recommendations are illustrative and are not connected to real suppliers or purchase orders.

---

## 📈 Streamlit Dashboard

The project includes an interactive dashboard built with Streamlit.

### Dashboard Sections

1. **Overview**
   - Key inventory and demand metrics
   - Demand trends
   - Summary insights

2. **Demand Forecast**
   - Seven-day demand forecast
   - Thirty-day demand forecast
   - Store and product-family filters

3. **Inventory & Reorder**
   - Current simulated stock
   - Reorder status
   - Recommended order quantities
   - Inventory insights

4. **Model Performance**
   - Model comparison
   - Evaluation metrics
   - Model performance visualizations

The dashboard uses precomputed forecast and inventory CSV files generated during model development.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Jupyter Notebook | Data analysis and model development |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning |
| Joblib | Model serialization |
| Streamlit | Interactive dashboard |
| Plotly | Interactive visualizations |
| Git & GitHub | Version control and project hosting |
| VS Code | Application development |

---

## ⚠️ Challenges Faced

- Handling a large dataset containing more than three million original records.
- Creating historical lag and rolling features for multiple store and product-family combinations.
- Maintaining chronological order during model validation to respect the time-series nature of the problem.
- Comparing models using multiple evaluation metrics rather than relying on a single score.
- Generating multi-step forecasts recursively, where earlier predictions become inputs for later predictions.
- Designing inventory recommendations despite the absence of actual stock information in the original dataset.
- Clearly distinguishing synthetic data from real historical observations.

---

## ▶️ How to Run Locally

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the Project Folder

```bash
cd Smart_Stock
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Environment

**Windows:**

```bash
.venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser.

**Note:** The dashboard requires the forecast, inventory, and model-results CSV files in the expected `artifacts/` directory. Keep these files in place when running the application.

---

## 📁 Project Structure

```text
Smart_Stock/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── Data/
│   ├── raw_data/
│   └── processed/
│
├── notebooks/
│   └── model_development.ipynb
│
├── artifacts/
│   ├── final_hgb_model.pkl
│   ├── weekly_forecast.csv
│   ├── monthly_forecast.csv
│   ├── inventory.csv
│   ├── final_model_results.csv
│   └── initial_model_comparison_2025.csv
│
└── images/
    └── dashboard.png
```

*Adjust the folder and notebook names to match the actual files in your repository.*

---

## 🚧 Limitations and Future Improvements

### Current Limitations

- The original dataset contains historical records only up to August 2017.
- The extended 2018–2026 records are synthetic and cannot represent actual current market conditions.
- Inventory quantities are simulated because actual stock-level data is unavailable.
- Future promotional activity is assumed to be zero because future promotion schedules are unknown.
- Recursive forecasting can accumulate prediction errors over longer forecast horizons.
- The system does not currently connect to live retail databases or supplier systems.

### Future Improvements

- Integrate real-time retail sales and inventory data.
- Incorporate actual inventory levels and supplier lead times.
- Include holiday calendars, oil prices, and transaction-related information where relevant.
- Explore additional forecasting models and advanced time-series techniques.
- Add explainable AI techniques such as SHAP.
- Develop a FastAPI backend for model inference and integration.
- Deploy the application to Streamlit Community Cloud or another hosting platform.

---

## 👩‍💻 Author

**Your Name**

- GitHub: [Your GitHub Profile](YOUR_GITHUB_PROFILE_URL)
- LinkedIn: [Your LinkedIn Profile](YOUR_LINKEDIN_URL)
- Email: YOUR_EMAIL_ADDRESS

---

## 📌 Project Summary

SmartStock demonstrates an end-to-end machine learning workflow, from retail sales data analysis and feature engineering to model comparison, demand forecasting, inventory recommendations, and interactive dashboard development.

The project aims to demonstrate how machine learning can support retail inventory planning and data-driven decision-making.


