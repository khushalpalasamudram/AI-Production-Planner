import streamlit as st
import pandas as pd
import numpy as np
import joblib


st.set_page_config(page_title="AI Production Planner", layout="wide")


# LOAD DATA + MODEL

df = pd.read_csv("final_merged_data.csv")
df['date'] = pd.to_datetime(df['date'])

rf_model = joblib.load("model.pkl")


st.title("AI Production Demand Aligner")
st.markdown("### Real-time Forecasting • Optimization • Explainability")


# SIDEBAR CONTROLS

st.sidebar.header("⚙ Scenario Controls")

products = df['product'].unique()
product = st.sidebar.selectbox("Select Product", products)

demand_factor = st.sidebar.slider("Demand Multiplier", 0.5, 1.5, 1.0)
material_factor = st.sidebar.slider("Material Availability", 0.5, 1.5, 1.0)
labor_factor = st.sidebar.slider("Labor Availability", 0.5, 1.5, 1.0)

# GET DATA

product_df = df[df['product'] == product].sort_values('date')
row = product_df.iloc[-1]
feature_columns = joblib.load("features.pkl")

# FEATURE BUILDING

input_data = {
    'lag_1': row['lag_1'],
    'lag_7': row['lag_7'],
    'month': row['month'],
    'day_of_week': row['day_of_week'],
    'effective_labor': row['labor_available'] * labor_factor,
    'effective_capacity': row['machine_capacity'],
    'effective_material': row['material_available'] * material_factor
}

for col in df.columns:
    if col.startswith("product_"):
        input_data[col] = 1 if col == f'product_{product}' else 0

input_df = pd.DataFrame(columns=feature_columns)

# Fill row
input_df.loc[0] = 0

# Add actual values
input_df.at[0, 'lag_1'] = row['lag_1']
input_df.at[0, 'lag_7'] = row['lag_7']
input_df.at[0, 'month'] = row['month']
input_df.at[0, 'day_of_week'] = row['day_of_week']

input_df.at[0, 'effective_labor'] = row['labor_available'] * labor_factor
input_df.at[0, 'effective_capacity'] = row['machine_capacity']
input_df.at[0, 'effective_material'] = row['material_available'] * material_factor

# Product encoding
product_col = f"product_{product}"
if product_col in input_df.columns:
    input_df.at[0, product_col] = 1

# PREDICTION

predicted_demand = rf_model.predict(input_df)[0] * demand_factor

# CONSTRAINTS

effective_material = input_data['effective_material']
effective_labor = input_data['effective_labor']
effective_capacity = input_data['effective_capacity']

production = min(
    predicted_demand,
    effective_material,
    effective_labor,
    effective_capacity
)


# BOTTLENECK DETECTION

constraints = {
    "Material": effective_material,
    "Labor": effective_labor,
    "Capacity": effective_capacity
}

min_value = min(constraints.values())

# Find all close constraints (within 10%)
bottlenecks = [
    k for k, v in constraints.items()
    if v <= min_value * 1.1
]

bottleneck = ", ".join(bottlenecks)

# EXPLANATION

reasons = []

if effective_material < predicted_demand:
    reasons.append(f"Material ({round(effective_material,1)})")

if effective_labor < predicted_demand:
    reasons.append(f"Labor ({round(effective_labor,1)})")

if effective_capacity < predicted_demand:
    reasons.append(f"Capacity ({round(effective_capacity,1)})")

if reasons:
    explanation = f"Demand {round(predicted_demand,1)} → Production {round(production,1)} limited by " + ", ".join(reasons)
else:
    explanation = f"Demand {round(predicted_demand,1)} fully met"

# SEVERITY
gap = predicted_demand - production

if gap > 100:
    severity = "High"
    color = "🔴"
elif gap > 50:
    severity = "Medium"
    color = "🟠"
else:
    severity = "Low"
    color = "🟢"

# RECOMMENDATIONS

recommendations = []

if "Labor" in bottlenecks:
    recommendations.append("Increase workforce or add shifts")

if "Material" in bottlenecks:
    recommendations.append("Improve supplier delivery or increase inventory")

if "Capacity" in bottlenecks:
    recommendations.append("Increase machine capacity or reduce downtime")

if gap > 100:
    recommendations.append("Consider demand prioritization or outsourcing")

# Display
st.subheader("💡 Recommendations")

for rec in recommendations:
    st.write(f"✔ {rec}")


# DISPLAY KPIs

col1, col2, col3, col4 = st.columns(4)

col1.metric("Predicted Demand", round(predicted_demand, 1))
col2.metric("Production Plan", round(production, 1))
col3.metric("Gap", round(gap, 1))
col4.metric("Bottleneck", bottleneck)

# EXPLANATION

st.subheader("🧠 AI Explanation")
st.success(explanation)

# SEVERITY

st.subheader("⚠ Risk Level")
st.write(f"{color} **{severity} Impact**")


# TREND VISUALIZATION

st.subheader("📈 Demand Trend (Last 30 Days)")

chart_data = product_df.tail(30)
st.line_chart(chart_data.set_index('date')['demand'])


# RAW DATA

with st.expander("View Latest Data"):
    st.dataframe(product_df.tail(5))
st.subheader("📊 Scenario Comparison")

st.write(f"Old Demand: {round(row['demand'],1)}")
st.write(f"New Demand: {round(predicted_demand,1)}")