
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(
    page_title="IndiGo Flight Recommendation System",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ IndiGo Airlines Recommendation System")
st.write("AI-powered flight prediction and recommendation using Decision Tree.")

@st.cache_data
def create_dataset():
    np.random.seed(42)

    cities = ["Delhi", "Mumbai", "Bengaluru", "Hyderabad",
              "Chennai", "Kolkata", "Jaipur", "Pune"]

    rows = []

    for i in range(2000):
        source, destination = np.random.choice(cities, 2, replace=False)
        price = np.random.randint(2500, 12001)
        duration = round(np.random.uniform(1, 5.5), 1)
        stops = np.random.choice([0, 1], p=[0.75, 0.25])
        purpose = np.random.choice(["Business", "Leisure", "Family"])
        budget = np.random.randint(3000, 13001)
        max_duration = round(np.random.uniform(1.5, 6), 1)

        suitable = int(
            price <= budget
            and duration <= max_duration
            and (stops == 0 or purpose != "Business")
        )

        rows.append([
            source, destination, price, duration,
            stops, purpose, budget, max_duration, suitable
        ])

    return pd.DataFrame(rows, columns=[
        "Source", "Destination", "Price", "Duration",
        "Stops", "Purpose", "Budget", "Max_Duration", "Suitable"
    ])

data = create_dataset()

categorical_features = [
    "Source", "Destination", "Purpose"
]

numeric_features = [
    "Price", "Duration", "Stops", "Budget", "Max_Duration"
]

preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"),
     categorical_features),
    ("numeric", StandardScaler(), numeric_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(
        max_depth=8,
        min_samples_leaf=5,
        random_state=42
    ))
])

X = data[categorical_features + numeric_features]
y = data["Suitable"]

model.fit(X, y)

st.sidebar.header("Passenger Preferences")

cities = sorted(data["Source"].unique())

source = st.sidebar.selectbox("Departure City", cities)
destination_options = [city for city in cities if city != source]
destination = st.sidebar.selectbox("Destination City", destination_options)

budget = st.sidebar.slider(
    "Maximum Budget (₹)", 3000, 13000, 7000, step=500
)

max_duration = st.sidebar.slider(
    "Maximum Flight Duration (hours)", 1.5, 6.0, 4.0, step=0.5
)

purpose = st.sidebar.selectbox(
    "Travel Purpose", ["Business", "Leisure", "Family"]
)

preferred_stops = st.sidebar.selectbox(
    "Preferred Flight Type",
    ["Any", "Non-stop only", "Connecting allowed"]
)

if st.sidebar.button("Recommend Flights", type="primary"):

    rng = np.random.default_rng(42)

    candidates = []

    for i in range(30):
        price = int(rng.integers(2500, 12001))
        duration = round(float(rng.uniform(1, 5.5)), 1)
        stops = int(rng.choice([0, 1], p=[0.75, 0.25]))

        if preferred_stops == "Non-stop only" and stops != 0:
            continue

        candidates.append({
            "Source": source,
            "Destination": destination,
            "Price": price,
            "Duration": duration,
            "Stops": stops,
            "Purpose": purpose,
            "Budget": budget,
            "Max_Duration": max_duration
        })

    if not candidates:
        st.warning("No candidate flights match your selected preferences.")
    else:
        flights = pd.DataFrame(candidates)

        flights["Prediction"] = model.predict(
            flights[categorical_features + numeric_features]
        )

        probabilities = model.predict_proba(
            flights[categorical_features + numeric_features]
        )

        suitable_index = list(model.named_steps["classifier"].classes_).index(1)
        flights["Suitability Score (%)"] = (
            probabilities[:, suitable_index] * 100
        ).round(1)

        recommendations = flights[
            flights["Prediction"] == 1
        ].copy()

        recommendations = recommendations.sort_values(
            by=["Suitability Score (%)", "Price"],
            ascending=[False, True]
        )

        st.subheader("Prediction Results")

        col1, col2, col3 = st.columns(3)

        col1.metric("Candidate Flights", len(flights))
        col2.metric("Suitable Flights", len(recommendations))
        col3.metric(
            "Lowest Recommended Fare",
            f"₹{recommendations['Price'].min():,}"
            if not recommendations.empty else "N/A"
        )

        if recommendations.empty:
            st.warning(
                "No suitable flights found. Try increasing your budget "
                "or allowing a longer flight duration."
            )
        else:
            st.success("Recommended flight options based on your preferences")

            display = recommendations.rename(columns={
                "Source": "From",
                "Destination": "To",
                "Price": "Estimated Fare (₹)",
                "Duration": "Duration (Hours)",
                "Stops": "Stops",
                "Suitability Score (%)": "Suitability (%)"
            })

            st.dataframe(
                display[[
                    "From", "To", "Estimated Fare (₹)",
                    "Duration (Hours)", "Stops", "Suitability (%)"
                ]],
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "These are simulated flight options, not live IndiGo fares "
                "or confirmed schedules. The suitability scores are model "
                "outputs, not actual booking probabilities."
            )

            st.download_button(
                "Download Recommendations CSV",
                recommendations.to_csv(index=False).encode("utf-8"),
                file_name="indigo_recommendations.csv",
                mime="text/csv"
            )

with st.expander("Model Information"):
    st.write("Algorithm: Decision Tree Classifier")
    st.write("Preprocessing: One-Hot Encoding and Standard Scaling")
    st.write("Architecture: Scikit-learn Pipeline")
    st.write("Recommendation: Filter suitable candidates and rank by score")
    st.write("Training dataset: 2,000 synthetically generated examples")
