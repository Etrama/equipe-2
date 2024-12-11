import numpy as np
import pickle
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import psycopg2
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="Siackathon Team 2: Prediction App",
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
def rf_model_iron():
    with open('models/rf_iron.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def rf_model_silicate():
    with open('models/rf_silicate.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


# Prediction function for the 'iron' model
def predict_iron(features):
    model = rf_model_iron()
    return model.predict(features)

# Prediction function for the 'silicate' model
def predict_silicate(features):
    # Load the model for 'silicate'
    model = rf_model_silicate()
    return model.predict(features)


# Load data and train model
data = generate_sample_data()
# model, scaler = train_model(data)


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

def db_health():
    st.title(" ⚗️ Database Health")

    st.info("Tables and null check.")

    host = st.text_input("Host", placeholder="hostname")
    port = st.text_input("Port", placeholder="portnb")
    database = st.text_input("Database", placeholder="mydatabase")
    username = st.text_input("Username", placeholder="myuser")
    password = st.text_input("Password", type="password")
    # Create a function to connect to the database and get the null counts
    def get_null_counts(host, port, database, username, password):
        try:
            # Connect to the database
            conn = psycopg2.connect(
                host=host,
                port=port,
                database=database,
                user=username,
                password=password
            )
            
            # Get the list of tables in the database
            cur = conn.cursor()
            cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public';")
            tables = [row[0] for row in cur.fetchall()]
            
            # Create a dictionary to store the null counts
            null_counts = {}
            
            # Loop through each table and get the null counts
            for table in tables:
                cur.execute(f"SELECT * FROM {table};")
                rows = cur.fetchall()
                columns = [desc[0] for desc in cur.description]
                df = pd.DataFrame(rows, columns=columns)
                null_counts[table] = df.isnull().sum().to_dict()
            
            # Close the connection
            conn.close()
            
            return null_counts
        
        except psycopg2.Error as e:
            st.error(f"Error connecting to the database: {e}")
            return None
    # Create a button to connect to the database and get the null counts
    if st.button("Connect and Get Null Counts"):
        null_counts = get_null_counts(host, port, database, username, password)
        
        if null_counts is not None:
            # Display the null counts
            st.header("Null Counts")
            for table, counts in null_counts.items():
                st.subheader(table)
                st.write(counts)

def prediction_page():
    st.title("🎯 Prediction")

    st.info("Enter values for each variable to get predictions for both models. Below are the statistical ranges for each variable based on historical data:")

    # Display the statistical summary for each variable
    st.subheader("Variable Statistics")

    # Input fields for each group of variables
    st.subheader("Group 1: Feed and Flow Parameters")
    col1, col2 = st.columns(2)

    # Define variables and their range (min, max, mean)
    variables_group_1 = {
        '% Iron Feed': (43.37, 64.37, 57.09),
        '% Silica Feed': (3.99, 32.23, 13.69),
        'Starch Flow': (79.12, 6157.48, 2814.47),
        'Amina Flow': (242.79, 736.51, 489.07),
        'Ore Pulp Flow': (377.06, 417.88, 400.11),
    }

    for var_name, (min_val, max_val, mean_val) in variables_group_1.items():
        with col1 if var_name in list(variables_group_1.keys())[:3] else col2:
            st.write(f"{var_name} (Min: {min_val}, Max: {max_val}, Mean: {mean_val:.2f})")
            st.session_state[var_name] = st.number_input(
                f"Enter value for {var_name}",
                value=mean_val,
                min_value=min_val,
                max_value=max_val,
                help=f"Input value for {var_name} within the specified range."
            )

    st.subheader("Group 2: Ore and Reagents Parameters")
    col3, col4 = st.columns(2)

    variables_group_2 = {
        'Ore Pulp pH': (8.77, 10.81, 9.82),
        'Ore Pulp Density': (1.52, 1.83, 1.68),
        'Cell Spin Factor': (0.69, 50848.50, 14722.06),
        'Bubble Size': (0.00, 50847.79, 14721.92),
        'Iron Binding Agent (kg)': (10.05, 35.33, 24.11),
        'Binder Activator (kg)': (10.05, 35.33, 24.56),
    }

    for var_name, (min_val, max_val, mean_val) in variables_group_2.items():
        with col3 if var_name in list(variables_group_2.keys())[:3] else col4:
            st.write(f"{var_name} (Min: {min_val}, Max: {max_val}, Mean: {mean_val:.2f})")
            st.session_state[var_name] = st.number_input(
                f"Enter value for {var_name}",
                value=mean_val,
                min_value=min_val,
                max_value=max_val,
                help=f"Input value for {var_name} within the specified range."
            )

    st.subheader("Group 3: Flotation Column Parameters")
    col5, col6 = st.columns(2)

    variables_group_3 = {
        'Flotation Column 01 Air Flow': (287.06, 306.40, 300.36),
        'Flotation Column 02 Air Flow': (196.52, 355.06, 285.85),
        'Flotation Column 03 Air Flow': (199.73, 350.61, 286.15),
        'Flotation Column 04 Air Flow': (182.05, 859.06, 528.06),
        'Flotation Column 05 Air Flow': (227.96, 824.35, 528.83),
        'Flotation Column 06 Air Flow': (135.21, 884.83, 535.82),
        'Flotation Column 07 Air Flow': (165.66, 675.64, 424.26),
        'Flotation Column 01 Level': (214.98, 674.07, 429.89),
        'Flotation Column 02 Level': (203.73, 698.51, 431.33),
        'Flotation Column 03 Level': (188.95, 655.50, 426.04),
        'Flotation Column 04 Level': (188.95, 655.50, 426.04),
        'Flotation Column 05 Level': (188.95, 655.50, 426.04),
        'Flotation Column 06 Level': (188.95, 655.50, 426.04),
        'Flotation Column 07 Level': (188.95, 655.50, 426.04),
    }

    for var_name, (min_val, max_val, mean_val) in variables_group_3.items():
        with col5 if var_name in list(variables_group_3.keys())[:6] else col6:
            st.write(f"{var_name} (Min: {min_val}, Max: {max_val}, Mean: {mean_val:.2f})")
            st.session_state[var_name] = st.number_input(
                f"Enter value for {var_name}",
                value=mean_val,
                min_value=min_val,
                max_value=max_val,
                help=f"Input value for {var_name} within the specified range."
            )

    # Prediction button
    if st.button("🚀 Predict", type="primary"):
        # Gather all inputs in a dictionary or dataframe
        features = pd.DataFrame({
            "% Iron Feed": [st.session_state["% Iron Feed"]],
            "% Silica Feed": [st.session_state["% Silica Feed"]],
            "Starch Flow": [st.session_state["Starch Flow"]],
            "Amina Flow": [st.session_state["Amina Flow"]],
            "Ore Pulp Flow": [st.session_state["Ore Pulp Flow"]],
            # Continue for all features...
        })

        # You can then make predictions for both models here using the `predict_iron` and `predict_silicate` functions.
        prediction_iron = predict_iron(features)
        prediction_silicate = predict_silicate(features)

        # Show results
        st.success(f"### Predicted Value for Iron Model: {prediction_iron[0]:.2f}")
        st.success(f"### Predicted Value for Silicate Model: {prediction_silicate[0]:.2f}")

def about_page():
    st.title("📖 Data Story Telling ;)")

    st.markdown(
        """
        ### Siackathon Team 2: Prediction App
        Welcome to the **Prediction App** developed by **Siackathon Team 2**! 
        This application allows you to visualize data, make predictions using machine learning, and explore trends in various variables over time.

        ### Features
        - **Dashboard**: Visualize the evolution of target variables with smoothing options.
        - **Prediction**: Input your own values for prediction using a trained model.

        ### Technologies Used
        - Streamlit
        - Plotly for data visualization
        - Scikit-learn for machine learning

        ## Team
        - **Salha Yahia**
        - **Mathieu Jacquand**
        - **Kaushik Moudgalya**

        ## Acknowledgements:
        - No time to make ppt so Data Storytelling will have to suffice :)
        
        ## Story:
        - The dataset is quite largish with over 700k records. We realized pretty early that we have some insidious data which we tried to fix.
        - The rationale behind this was that we would lose over 3% (1.1% float, 1.9% categorical) of our data if we simply dropped it and we didn't want to do that. Maybe we should have in hindsight :p
        - We found that we had data from the year 1824 lol, which was funny. We removed these records ofc.
        """
    )
    st.image('./images/missing data percentage.png', caption='Missing Data Pct', use_container_width=True)
    st.image('./images/missing_data.png', caption='Missing Data Example', use_container_width=True)
    st.markdown(
        """
        - We then looked at the distrbution of the 4 target variables we were provided.
        - The variables are NOT normally distributed and are pretty skewed.
        - HOWEVER! We decided NOT to standardize the data as Salha had worked on a similar project before and we need to output real values at the end and not standardized values.
        """)
    st.image('./images/target_dist1.png', caption='Distribution of Target Variables', use_container_width=True)
    st.markdown(
        """
        - We then looked at the distriubtion of the variables to see if we could find some outliers.
        - We can clearly see that Cell Spin Factor and Bubble Size (ficitional variables) heavily favor just one value.
        - The Flotation COlumns XX Air Flow are also interesting to look at as they seem to have a very discrete distribution.
        - We will deal with their discrete nature if we have the time.
        """)
    st.image('./images/var_dist.png', caption='Distribution of Variables', use_container_width=True)
    st.markdown(
        """
        - We then converted the categorical variables into numerical ones.
        - Below are some patterns that we noticed:
        """)
    st.image('./images/cat variable clean.png', caption='AIR Error', use_container_width=True)
    st.markdown("""
        - Other than missing values we also noticed that there were some errors in the data.""")
    st.image('./images/ERROR.png', caption='AIR Error', use_container_width=True)
    st.markdown("""
        - After dealing with all this we looked at the correlation between the variables.""")
    st.image('./images/corr.png', caption='Correlation Matrix', use_container_width=True)
    st.markdown("""
        - At this point we ran into a lot of deployment issues :(
        - We had to use some workaround to try and get it to work.
        - To be a little fancy we tried to mutlivariate imputation to replace the missing values.
        - This ensures that we lose no data and we can rely on the other variables to populate the missing data.
    """)
    st.image('./images/imputation.png', caption='Multivariate Imputation', use_container_width=True)
    st.markdown("""
    - Some metrics based on the models we trained:
    
    | **Metric** | **Silicate (Gradient Boosting)** | **Iron (Random Forest)** |
    |------------|----------------------------------|--------------------------|
    | **MAE**    | 0.844995479790331               | 1.3974971141781678       |
    | **MAPE**   | 0.354705627497454               | 0.022798151402587872     |

    - The Iron models are amazing.
    - The silicate models could use some more work.
    """)
    st.markdown("""
    - We also tried to do some feature importance analysis.
    - Feature importance of the iron model:
    """)
    st.image('./images/iron_feat_imp.png', caption='Feature Importance FE', use_container_width=True)
    st.markdown("""
    - Feature importance of the silicate model:
    """)
    st.image('./images/si_feat_imp.png', caption='Feature Importance Silicate', use_container_width=True)
    st.markdown("""
        - We also have a trend analysis of the predicted variables for both the models against the originals"""
    )
    st.image('./images/iron_trend.png', caption='FE Trend Analysis', use_container_width=True)
    st.image('./images/silicon_trend.png', caption='SI Trend Analysis', use_container_width=True)
    

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["Dashboard", "Prediction", "Data Story", "DB Health"], index=2)

# Display selected page
if page == "Dashboard":
    dashboard_page()
elif page == "Prediction":
    prediction_page()
elif page == "DB Health":
    db_health()
else:
    about_page()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    """
    Application developed by [Siackathon Team 2]
    """
)
