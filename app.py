import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="Prediction App",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Generate sample data
def generate_sample_data():
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")

    np.random.seed(42)
    # Generate smoother base signals using sine waves
    t = np.linspace(0, 4 * np.pi, len(dates))
    data = pd.DataFrame(
        {
            "date": dates,
            "var1": 50 + 10 * np.sin(t) + np.random.normal(0, 2, len(dates)),
            "var2": 100
            + 20 * np.sin(t + np.pi / 4)
            + np.random.normal(0, 4, len(dates)),
            "var3": 500
            + 50 * np.sin(t + np.pi / 3)
            + np.random.normal(0, 10, len(dates)),
            "var4": 25
            + 5 * np.sin(t + np.pi / 6)
            + np.random.normal(0, 1, len(dates)),
        }
    )

    # Create target variable with smooth trend and seasonal pattern
    data["target"] = (
        0.3 * data["var1"]
        + 0.2 * data["var2"]
        + 0.1 * data["var3"]
        + 0.15 * data["var4"]
        + 30 * np.sin(t / 2)  # Add seasonal pattern
        + np.random.normal(0, 5, len(dates))
    )  # Add some noise

    return data


# Create and train model
def train_model(data):
    X = data[["var1", "var2", "var3", "var4"]]
    y = data["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    return model, scaler


# Prediction function
def predict(model, scaler, features):
    features_scaled = scaler.transform(features)
    return model.predict(features_scaled)


# Load data and train model
data = generate_sample_data()
model, scaler = train_model(data)


def dashboard_page():
    st.title("📊 Dashboard")

    # Add smoothing parameter
    st.sidebar.markdown("### Chart Settings")
    window_size = st.sidebar.slider(
        "Smoothing window (days)",
        min_value=1,
        max_value=30,
        value=7,
        help="Adjust the smoothing level of the chart",
    )

    # Calculate smooth target
    smooth_target = (
        data["target"]
        .rolling(window=window_size, center=True, min_periods=1)
        .mean()
    )

    # Time series plot with both raw and smoothed data
    st.subheader("Target Variable Evolution")

    fig = go.Figure()

    # Add raw data as light scatter
    fig.add_trace(
        go.Scatter(
            x=data["date"],
            y=data["target"],
            mode="lines",
            name="Raw Data",
            line=dict(color="rgba(0,176,246,0.2)", width=1),
            showlegend=True,
        )
    )

    # Add smoothed line
    fig.add_trace(
        go.Scatter(
            x=data["date"],
            y=smooth_target,
            mode="lines",
            name="Smoothed Trend",
            line=dict(color="rgb(0,100,180)", width=3),
            showlegend=True,
        )
    )

    fig.update_layout(
        title="Time Series of Target Variable",
        xaxis_title="Date",
        yaxis_title="Target Value",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Statistics using smoothed data
    st.subheader("📈 Statistical Summary")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Average", f"{smooth_target.mean():.2f}")
        st.metric("Std Dev", f"{smooth_target.std():.2f}")

    with col2:
        st.metric("Max", f"{smooth_target.max():.2f}")
        st.metric("Min", f"{smooth_target.min():.2f}")


def prediction_page():
    st.title("🎯 Prediction")

    st.info("Enter values for each variable to get a prediction.")

    # Input fields
    col1, col2 = st.columns(2)

    with col1:
        var1 = st.number_input(
            "Variable 1",
            value=50.0,
            min_value=0.0,
            max_value=100.0,
            help="Enter value for Variable 1",
        )

        var2 = st.number_input(
            "Variable 2",
            value=100.0,
            min_value=0.0,
            max_value=200.0,
            help="Enter value for Variable 2",
        )

    with col2:
        var3 = st.number_input(
            "Variable 3",
            value=500.0,
            min_value=300.0,
            max_value=700.0,
            help="Enter value for Variable 3",
        )

        var4 = st.number_input(
            "Variable 4",
            value=25.0,
            min_value=0.0,
            max_value=50.0,
            help="Enter value for Variable 4",
        )

    # Prediction button
    if st.button("🚀 Predict", type="primary"):
        # Prepare features
        features = pd.DataFrame(
            {"var1": [var1], "var2": [var2], "var3": [var3], "var4": [var4]}
        )

        # Make prediction
        prediction = predict(model, scaler, features)[0]

        # Show results
        st.success(f"### Predicted Value: {prediction:.2f}")

        # Input summary
        st.subheader("Input Values:")
        st.dataframe(features, use_container_width=True)


# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["Dashboard", "Prediction"])

# Display selected page
if page == "Dashboard":
    dashboard_page()
else:
    prediction_page()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    """
    This is a demo application
    """
)
