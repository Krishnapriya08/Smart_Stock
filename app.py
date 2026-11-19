
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SmartStock",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

WEEKLY_PATH = ARTIFACTS_DIR / "weekly_forecast.csv"
MONTHLY_PATH = ARTIFACTS_DIR / "monthly_forecast.csv"
INVENTORY_PATH = ARTIFACTS_DIR / "inventory.csv"
MODEL_RESULTS_PATH = ARTIFACTS_DIR / "final_model_results.csv"
MODEL_PATH = ARTIFACTS_DIR / "final_hgb_model.pkl"

# ============================================================
# 2. COLOR THEME
# ============================================================

NAVY = "#102A43"
DARK_BLUE = "#163A5F"
BLUE = "#2471A3"
LIGHT_BLUE = "#EAF2F8"
BACKGROUND = "#F4F7FA"
WHITE = "#FFFFFF"
TEXT = "#243B53"
MUTED = "#627D98"
GREEN = "#238636"
RED = "#C0392B"
AMBER = "#D68910"

# ============================================================
# 3. CUSTOM CSS
# ============================================================

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {BACKGROUND};
        color: {TEXT};
    }}

    [data-testid="stSidebar"] {{
        background-color: {NAVY};
    }}

    [data-testid="stSidebar"] * {{
        color: #FFFFFF;
    }}

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
        color: #D9E6F2;
    }}

    [data-testid="stSidebar"] .stRadio label {{
        color: #FFFFFF !important;
    }}

    [data-testid="stSidebar"] hr {{
        border-color: #42627F;
    }}

    h1, h2, h3 {{
        color: {NAVY} !important;
        font-weight: 700;
    }}

    .main-title {{
        color: {NAVY};
        font-size: 32px;
        font-weight: 800;
        margin-bottom: 4px;
    }}

    .subtitle {{
        color: {MUTED};
        font-size: 14px;
        margin-bottom: 22px;
    }}

    .metric-card {{
        background-color: {WHITE};
        border: 1px solid #D9E2EC;
        border-radius: 10px;
        padding: 18px 20px;
        min-height: 112px;
        box-shadow: 0 2px 5px rgba(16,42,67,0.04);
    }}

    .metric-label {{
        color: {MUTED};
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 8px;
    }}

    .metric-value {{
        color: {NAVY};
        font-size: 27px;
        font-weight: 800;
    }}

    .metric-caption {{
        color: {MUTED};
        font-size: 11px;
        margin-top: 5px;
    }}

    .section-heading {{
        color: {NAVY};
        font-size: 19px;
        font-weight: 700;
        margin-top: 18px;
        margin-bottom: 12px;
    }}

    .info-card {{
        background-color: {WHITE};
        border: 1px solid #D9E2EC;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 12px;
    }}

    .status-good {{
        color: {GREEN};
        font-weight: 700;
    }}

    .status-warning {{
        color: {RED};
        font-weight: 700;
    }}

    div[data-testid="stDataFrame"] {{
        border: 1px solid #D9E2EC;
        border-radius: 8px;
    }}

    .stButton button {{
        background-color: {NAVY};
        color: white;
        border-radius: 7px;
        border: none;
    }}

    .stButton button:hover {{
        background-color: {DARK_BLUE};
        color: white;
    }}

    footer {{
        visibility: hidden;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# 4. DATA LOADING
# ============================================================

@st.cache_data
def load_csv(path):
    if not path.exists():
        return None

    df = pd.read_csv(path)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df


weekly_df = load_csv(WEEKLY_PATH)
monthly_df = load_csv(MONTHLY_PATH)
inventory_df = load_csv(INVENTORY_PATH)
model_results_df = load_csv(MODEL_RESULTS_PATH)

# ============================================================
# 5. DATA VALIDATION
# ============================================================

required_forecast_columns = {
    "date", "store_nbr", "family", "predicted_sales"
}

required_inventory_columns = {
    "store_nbr",
    "family",
    "average_daily_demand",
    "current_stock",
    "safety_stock",
    "reorder_point",
    "target_stock",
    "recommended_order_qty",
    "stock_status"
}

if weekly_df is not None:
    if not required_forecast_columns.issubset(weekly_df.columns):
        st.error("The weekly forecast file has an unexpected structure.")
        st.stop()

if monthly_df is not None:
    if not required_forecast_columns.issubset(monthly_df.columns):
        st.error("The monthly forecast file has an unexpected structure.")
        st.stop()

if inventory_df is not None:
    if not required_inventory_columns.issubset(inventory_df.columns):
        st.error("The inventory file has an unexpected structure.")
        st.stop()

missing_files = []

for path in [
    WEEKLY_PATH,
    MONTHLY_PATH,
    INVENTORY_PATH
]:
    if not path.exists():
        missing_files.append(path.name)

if missing_files:
    st.error(
        "Required project files are missing from the artifacts folder: "
        + ", ".join(missing_files)
    )
    st.info("Check that your CSV files are saved in the artifacts folder.")
    st.stop()

# ============================================================
# 6. COMMON FUNCTIONS
# ============================================================

def page_header(title, subtitle):
    st.markdown(
        f'<div class="main-title">{title}</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="subtitle">{subtitle}</div>',
        unsafe_allow_html=True
    )


def metric_card(label, value, caption=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def section_heading(title):
    st.markdown(
        f'<div class="section-heading">{title}</div>',
        unsafe_allow_html=True
    )


def format_number(value):
    return f"{value:,.0f}"


def chart_layout(fig, height=350):
    fig.update_layout(
        height=height,
        plot_bgcolor=WHITE,
        paper_bgcolor=WHITE,
        font=dict(
            family="Arial",
            color=TEXT,
            size=12
        ),
        margin=dict(l=20, r=20, t=35, b=20),
        legend_title_text=""
    )
    return fig


def forecast_insight(df):
    if df.empty:
        return "No forecast records are available for the selected filters."

    total = df["predicted_sales"].sum()
    highest = df.loc[df["predicted_sales"].idxmax()]

    return (
        f"The total predicted demand is {format_number(total)} units. "
        f"The highest individual forecast is for "
        f"{highest['family']} at Store {int(highest['store_nbr'])}."
    )


# ============================================================
# 7. SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 10px 0 20px 0;">
            <div style="font-size: 29px; font-weight: 800;">
                📦 SmartStock
            </div>
            <div style="font-size: 12px; color: #D9E6F2;">
                Inventory Intelligence System
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    selected_page = st.radio(
        "NAVIGATION",
        [
            "Home",
            "Dashboard",
            "Demand Forecast",
            "Inventory Management",
            "ML Model",
            # "About"
        ],
        label_visibility="visible"
    )

    st.divider()

    # st.markdown(
    #     """
    #     <div style="font-size: 11px; color: #D9E6F2;">
    #         SMART INVENTORY AND DEMAND PREDICTION SYSTEM
    #     </div>
    #     """,
    #     unsafe_allow_html=True
    # )


# ============================================================
# 8. HOME PAGE
# ============================================================

if selected_page == "Home":

    # page_header(
    #     "SmartStock",
    #     "Smart Inventory and Demand Prediction System"
    # )

    st.markdown(
        f"""
        <div style="
            background-color: {NAVY};
            padding: 35px;
            border-radius: 12px;
            color: white;
            margin-bottom: 25px;
        ">
            <div style="font-size: 29px; font-weight: 800; color: white;">
                SmartStock
            </div>
            <p style="color: #D9E6F2; font-size: 15px; margin-top: 12px;">
                An intelligent system that predicts product demand
                and provides inventory replenishment recommendations
                using machine learning.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### System Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="info-card">
                <h3>Demand Forecasting</h3>
                <p>Generate 7-day and 30-day demand forecasts
                for stores and product families.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="info-card">
                <h3>Inventory Monitoring</h3>
                <p>Monitor current stock levels and identify
                products that require replenishment.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="info-card">
                <h3>Smart Reordering</h3>
                <p>Calculate reorder points and recommend
                suitable order quantities.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    section_heading("Project Highlights")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Stores", "54", "Retail locations")

    with c2:
        metric_card("Product Families", "33", "Product categories")

    with c3:
        metric_card("Forecast Periods", "7 & 30 Days", "Weekly and monthly")

    with c4:
        metric_card("ML Algorithm", "HGB", "HistGradientBoosting")

    # section_heading("How SmartStock Works")

    # st.markdown(
    #     """
    #     1. **Data Preparation:** Historical sales data is cleaned
    #        and transformed into useful model features.
    #     2. **Model Training:** Machine learning models are evaluated
    #        and the best-performing model is selected.
    #     3. **Demand Forecasting:** Future demand is predicted
    #        for each store and product family.
    #     4. **Inventory Analysis:** Current stock is compared
    #        against calculated reorder points.
    #     5. **Reorder Recommendation:** The system calculates
    #        the recommended quantity to replenish.
    #     """
    # )


# ============================================================
# 9. DASHBOARD
# ============================================================

elif selected_page == "Dashboard":

    page_header(
        "Dashboard",
        "Overview of demand forecasts and inventory status"
    )

    total_stores = inventory_df["store_nbr"].nunique()
    total_families = inventory_df["family"].nunique()
    total_records = len(inventory_df)

    reorder_count = (
        inventory_df["stock_status"] == "REORDER NOW"
    ).sum()

    sufficient_count = (
        inventory_df["stock_status"] == "SUFFICIENT STOCK"
    ).sum()

    total_stock = inventory_df["current_stock"].sum()
    total_order = inventory_df["recommended_order_qty"].sum()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Total Stores", format_number(total_stores))

    with c2:
        metric_card("Product Families", format_number(total_families))

    with c3:
        metric_card("Current Stock", format_number(total_stock), "Simulated units")

    with c4:
        metric_card("Reorder Alerts", format_number(reorder_count), "Store-family records")

    section_heading("Inventory Status Overview")

    left, right = st.columns(2)

    with left:
        status_data = pd.DataFrame({
            "Status": ["Sufficient Stock", "Reorder Now"],
            "Count": [sufficient_count, reorder_count]
        })

        fig = px.pie(
            status_data,
            names="Status",
            values="Count",
            hole=0.60,
            color="Status",
            color_discrete_map={
                "Sufficient Stock": GREEN,
                "Reorder Now": RED
            }
        )

        fig.update_traces(
            textposition="inside",
            textinfo="label+percent"
        )

        st.plotly_chart(chart_layout(fig), use_container_width=True)

        st.caption(
            "Most store-family combinations currently have sufficient "
            "stock according to the simulated inventory records."
        )

    with right:
        family_orders = (
            inventory_df.groupby("family")["recommended_order_qty"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        fig = px.bar(
            family_orders.sort_values("recommended_order_qty"),
            x="recommended_order_qty",
            y="family",
            orientation="h",
            color_discrete_sequence=[BLUE],
            labels={
                "recommended_order_qty": "Recommended Order Quantity",
                "family": "Product Family"
            }
        )

        st.plotly_chart(chart_layout(fig), use_container_width=True)

        st.caption(
            "The chart displays the ten product families with the "
            "largest total recommended order quantities."
        )

    section_heading("Demand Forecast Overview")

    forecast_type = st.selectbox(
        "Select forecast period",
        ["Weekly Forecast", "Monthly Forecast"]
    )

    selected_forecast = (
        weekly_df
        if forecast_type == "Weekly Forecast"
        else monthly_df
    )

    daily_summary = (
        selected_forecast.groupby("date")["predicted_sales"]
        .sum()
        .reset_index()
    )

    fig = px.line(
        daily_summary,
        x="date",
        y="predicted_sales",
        markers=True,
        color_discrete_sequence=[BLUE],
        labels={
            "date": "Date",
            "predicted_sales": "Predicted Demand"
        }
    )

    st.plotly_chart(chart_layout(fig, 370), use_container_width=True)

    st.caption(
        "Daily predicted demand aggregated across all stores "
        "and product families."
    )

    section_heading("Key Inventory Insights")

    st.markdown(
        f"""
        - **{reorder_count:,}** store-family combinations require
          replenishment.
        - **{sufficient_count:,}** combinations currently have
          sufficient stock.
        - The total recommended order quantity is
          **{total_order:,.0f} units**.
        """
    )


# ============================================================
# 10. DEMAND FORECAST PAGE
# ============================================================

elif selected_page == "Demand Forecast":

    page_header(
        "Demand Forecast",
        "Explore predicted demand across stores and product families"
    )

    forecast_choice = st.radio(
        "Forecast Period",
        ["7-Day Forecast", "30-Day Forecast"],
        horizontal=True
    )

    forecast_df = (
        weekly_df.copy()
        if forecast_choice == "7-Day Forecast"
        else monthly_df.copy()
    )

    st.markdown(
        f"**Forecast period:** "
        f"{forecast_df['date'].min().strftime('%d %b %Y')} "
        f"to {forecast_df['date'].max().strftime('%d %b %Y')}"
    )

    filter1, filter2 = st.columns(2)

    with filter1:
        store_options = ["All Stores"] + sorted(
            forecast_df["store_nbr"].unique().tolist()
        )

        selected_store = st.selectbox(
            "Select Store",
            store_options
        )

    with filter2:
        family_options = ["All Families"] + sorted(
            forecast_df["family"].unique().tolist()
        )

        selected_family = st.selectbox(
            "Select Product Family",
            family_options
        )

    filtered_forecast = forecast_df.copy()

    if selected_store != "All Stores":
        filtered_forecast = filtered_forecast[
            filtered_forecast["store_nbr"] == selected_store
        ]

    if selected_family != "All Families":
        filtered_forecast = filtered_forecast[
            filtered_forecast["family"] == selected_family
        ]

    total_predicted = filtered_forecast["predicted_sales"].sum()
    average_daily = (
        filtered_forecast.groupby("date")["predicted_sales"]
        .sum()
        .mean()
    )

    fc1, fc2, fc3 = st.columns(3)

    with fc1:
        metric_card(
            "Total Predicted Demand",
            format_number(total_predicted),
            "Units for selected filters"
        )

    with fc2:
        metric_card(
            "Average Daily Demand",
            format_number(average_daily),
            "Units per day"
        )

    with fc3:
        metric_card(
            "Forecast Records",
            format_number(len(filtered_forecast))
        )

    section_heading("Predicted Demand Trend")

    daily_forecast = (
        filtered_forecast.groupby("date")["predicted_sales"]
        .sum()
        .reset_index()
    )

    fig = px.line(
        daily_forecast,
        x="date",
        y="predicted_sales",
        markers=True,
        color_discrete_sequence=[BLUE],
        labels={
            "date": "Date",
            "predicted_sales": "Predicted Demand"
        }
    )

    fig.update_traces(line=dict(width=3))

    st.plotly_chart(chart_layout(fig, 400), use_container_width=True)

    st.info(forecast_insight(filtered_forecast))

    section_heading("Forecast Data")

    display_forecast = filtered_forecast.copy()
    display_forecast["date"] = display_forecast["date"].dt.strftime(
        "%Y-%m-%d"
    )

    st.dataframe(
        display_forecast,
        use_container_width=True,
        hide_index=True
    )

    csv_data = filtered_forecast.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Forecast CSV",
        data=csv_data,
        file_name=(
            "weekly_forecast.csv"
            if forecast_choice == "7-Day Forecast"
            else "monthly_forecast.csv"
        ),
        mime="text/csv"
    )

    # st.caption(
    #     "Future promotions are assumed to be zero because future "
    #     "promotion schedules are not available."
    # )


# ============================================================
# 11. INVENTORY MANAGEMENT PAGE
# ============================================================

elif selected_page == "Inventory Management":

    page_header(
        "Inventory Management",
        "Monitor stock levels and manage replenishment recommendations"
    )

    total_stock = inventory_df["current_stock"].sum()
    reorder_count = (
        inventory_df["stock_status"] == "REORDER NOW"
    ).sum()

    sufficient_count = (
        inventory_df["stock_status"] == "SUFFICIENT STOCK"
    ).sum()

    total_order = inventory_df["recommended_order_qty"].sum()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Total Current Stock", format_number(total_stock))

    with c2:
        metric_card("Reorder Alerts", format_number(reorder_count))

    with c3:
        metric_card("Sufficient Stock", format_number(sufficient_count))

    with c4:
        metric_card("Recommended Order", format_number(total_order), "Units")

    section_heading("Inventory Filters")

    filter1, filter2 = st.columns(2)

    with filter1:
        store_options = ["All Stores"] + sorted(
            inventory_df["store_nbr"].unique().tolist()
        )

        inventory_store = st.selectbox(
            "Select Store",
            store_options
        )

    with filter2:
        family_options = ["All Families"] + sorted(
            inventory_df["family"].unique().tolist()
        )

        inventory_family = st.selectbox(
            "Select Product Family",
            family_options
        )

    status_filter = st.selectbox(
        "Stock Status",
        [
            "All",
            "REORDER NOW",
            "SUFFICIENT STOCK"
        ]
    )

    filtered_inventory = inventory_df.copy()

    if inventory_store != "All Stores":
        filtered_inventory = filtered_inventory[
            filtered_inventory["store_nbr"] == inventory_store
        ]

    if inventory_family != "All Families":
        filtered_inventory = filtered_inventory[
            filtered_inventory["family"] == inventory_family
        ]

    if status_filter != "All":
        filtered_inventory = filtered_inventory[
            filtered_inventory["stock_status"] == status_filter
        ]

    section_heading("Inventory Records")

    st.dataframe(
        filtered_inventory,
        use_container_width=True,
        hide_index=True
    )

    inventory_csv = filtered_inventory.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Inventory Report",
        data=inventory_csv,
        file_name="inventory_report.csv",
        mime="text/csv"
    )

    section_heading("Reorder Recommendations")

    reorder_df = filtered_inventory[
        filtered_inventory["stock_status"] == "REORDER NOW"
    ].copy()

    if reorder_df.empty:
        st.success("No reorder alerts for the selected filters.")

    else:
        reorder_df = reorder_df.sort_values(
            "recommended_order_qty",
            ascending=False
        )

        st.warning(
            f"{len(reorder_df):,} records require replenishment "
            "under the selected filters."
        )

        st.dataframe(
            reorder_df[
                [
                    "store_nbr",
                    "family",
                    "current_stock",
                    "average_daily_demand",
                    "reorder_point",
                    "recommended_order_qty"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    section_heading("Recommended Order Quantity by Product Family")

    order_summary = (
        filtered_inventory.groupby("family")[
            "recommended_order_qty"
        ]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig = px.bar(
        order_summary.sort_values("recommended_order_qty"),
        x="recommended_order_qty",
        y="family",
        orientation="h",
        color_discrete_sequence=[BLUE],
        labels={
            "recommended_order_qty": "Recommended Order Quantity",
            "family": "Product Family"
        }
    )

    st.plotly_chart(chart_layout(fig, 400), use_container_width=True)

    st.caption(
        "Reorder point uses three days of lead-time demand plus "
        "two days of safety stock. Target stock covers seven days "
        "of demand plus safety stock."
    )


# ============================================================
# 12. ML MODEL PAGE
# ============================================================

elif selected_page == "ML Model":

    page_header(
        "Machine Learning Model",
        "Model evaluation, performance comparison and feature importance"
    )

    st.markdown("### Selected Model")

    st.markdown(
        f"""
        <div class="info-card">
            <h3>HistGradientBoosting Regressor</h3>
            <p>
                The final model was selected after comparing
                Decision Tree, Random Forest and
                HistGradientBoosting models using chronological
                validation.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Final model evaluation metrics
    final_metrics = {
        "MAE": 101.8667,
        "RMSE": 367.2787,
        "R2 Score": 0.8814
    }

    if model_results_df is not None and not model_results_df.empty:
        row = model_results_df.iloc[0]

        for key in ["MAE", "RMSE", "R2 Score"]:
            if key in model_results_df.columns:
                final_metrics[key] = float(row[key])

    section_heading("Final Model Test Performance")

    m1, m2, m3 = st.columns(3)

    with m1:
        metric_card("MAE", f"{final_metrics['MAE']:.2f}")

    with m2:
        metric_card("RMSE", f"{final_metrics['RMSE']:.2f}")

    with m3:
        metric_card("R² Score", f"{final_metrics['R2 Score']:.4f}")

    st.caption(
        "These metrics are from evaluation on the reserved 2026 "
        "synthetic test data."
    )

    section_heading("Chronological Cross-Validation Comparison")

    comparison_data = pd.DataFrame({
        "Model": [
            "Decision Tree",
            "Random Forest",
            "HistGradientBoosting"
        ],
        "Mean MAE": [
            120.1695,
            107.8620,
            108.0000
        ],
        "Mean RMSE": [
            463.7218,
            417.0334,
            395.7299
        ],
        "Mean R²": [
            0.8243,
            0.8579,
            0.8720
        ],
        "Mean Training Time (s)": [
            8.2143,
            37.7082,
            3.3714
        ]
    })

    st.dataframe(
        comparison_data,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Four expanding chronological validation folds covering "
        "2021–2024. Validation data in this experiment is synthetic."
    )

    section_heading("Model Performance Comparison")

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="MAE",
            x=comparison_data["Model"],
            y=comparison_data["Mean MAE"]
        )
    )

    fig.add_trace(
        go.Bar(
            name="RMSE",
            x=comparison_data["Model"],
            y=comparison_data["Mean RMSE"]
        )
    )

    fig.update_layout(
        barmode="group",
        yaxis_title="Error",
        xaxis_title="Model"
    )

    st.plotly_chart(chart_layout(fig, 400), use_container_width=True)

    section_heading("Feature Importance")

    feature_importance = pd.DataFrame({
        "Feature": [
            "rolling_mean_7",
            "onpromotion",
            "lag_14",
            "lag_7",
            "lag_1",
            "family",
            "day_of_week",
            "store_nbr",
            "month",
            "day",
            "year"
        ],
        "Importance": [
            269.0678,
            59.2513,
            34.7239,
            17.5948,
            16.8411,
            15.5596,
            12.8498,
            4.8182,
            0.6290,
            0.1155,
            0.0000
        ]
    })

    fig = px.bar(
        feature_importance.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        color_discrete_sequence=[BLUE],
        labels={
            "Importance": "Permutation Importance",
            "Feature": "Feature"
        }
    )

    st.plotly_chart(chart_layout(fig, 430), use_container_width=True)

    st.info(
        "The 7-day rolling mean has the highest permutation importance, "
        "followed by promotional activity. Importance measures predictive "
        "contribution, not causation."
    )

    section_heading("Model Configuration")

    model_parameters = pd.DataFrame({
        "Parameter": [
            "max_iter",
            "learning_rate",
            "max_leaf_nodes",
            "min_samples_leaf",
            "l2_regularization"
        ],
        "Value": [
            200,
            0.05,
            63,
            20,
            2.0
        ]
    })

    st.dataframe(
        model_parameters,
        use_container_width=True,
        hide_index=True
    )

    # if MODEL_PATH.exists():
    #     st.success("Saved final model file is available.")
    # else:
    #     st.warning(
    #         "The saved model file was not found. "
    #         "Model performance information is still displayed."
    #     )


# ============================================================
# 13. ABOUT PAGE
# ============================================================

# elif selected_page == "About":

#     page_header(
#         "About SmartStock",
#         "Project information and system details"
#     )

#     st.markdown(
#         """
        ### Project Description

    #     SmartStock is a machine-learning-based inventory and demand
    #     prediction system designed to support retail inventory planning.

    #     It uses historical sales information to forecast future demand
    #     and provides stock replenishment recommendations based on
    #     simulated inventory data.

    #     ### Technologies Used

    #     - Python
    #     - Pandas and NumPy
    #     - Scikit-learn
    #     - Streamlit
    #     - Plotly
    #     - Joblib
    #     - Jupyter Notebook
    #     - Visual Studio Code

    #     ### Machine Learning

    #     HistGradientBoosting Regressor is used as the final demand
    #     prediction model after model comparison and chronological
    #     validation.

    #     ### Dataset

    #     The project uses the Kaggle Store Sales – Time Series
    #     Forecasting dataset, which contains retail sales information
    #     from stores in Ecuador.

    #     ### Important Limitations

    #     - The original historical sales data ends in August 2017.
    #     - Data generated for later years is synthetic and does not
    #       represent actual recorded retail sales.
    #     - Future promotional activity is assumed to be zero.
    #     - Current stock quantities are simulated for demonstration.
    #     - Forecast accuracy may decrease over longer prediction horizons.
    #     """
    # )


# ============================================================
# 14. FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    f"""
    <div style="
        text-align: center;
        color: {MUTED};
        font-size: 12px;
        padding: 8px;
    ">
        SmartStock | Smart Inventory and Demand Prediction System
    </div>
    """,
    unsafe_allow_html=True
)
