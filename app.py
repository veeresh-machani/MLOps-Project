from plotly.graph_objs.layout import yaxis
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px  # interactive visualizations
import requests

# if st.checkbox("Show dataframe"):
#     chart_data = pd.DataFrame(np.random.randn(20, 4), columns=["a", "b", "c", "d"])

#     chart_data

st.set_page_config(page_title="Iris Classification", page_icon="🌸", layout="centered")


st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
    }
    h1, h2{
        color: #000000;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<h1 style='text-align: center; color: #000000;'> 🌸 Iris Classification 🌸 </h1>",
    unsafe_allow_html=True,
)
st.write("This is a simple iris classification app built with Streamlit and Python.")


with st.expander("About this model", expanded=False):
    st.write(
        """
        This is a simple iris classification model built with Streamlit and Python."
        The model is trained on the iris dataset and is able to classify the species of the iris flower.
        The model is a Random Forest classifier with 100 trees.
        The model is able to classify the species of the iris flower with an accuracy of 95%.
        The model is able to classify the species of the iris flower with an accuracy of 95%.
    """
    )
st.markdown("## Enter flower measurements")

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider(
        "🍃 Sepal Length (cm)", value=5.1, min_value=0.0, max_value=10.0, step=0.1
    )
    petal_length = st.slider(
        "🌺 Petal Length (cm)", value=1.4, min_value=0.0, max_value=10.0, step=0.1
    )
with col2:
    sepal_width = st.slider(
        "🍃 Sepal Width (cm)", value=3.5, min_value=0.0, max_value=10.0, step=0.1
    )
    petal_width = st.slider(
        "🌺 Petal Width (cm)", value=0.2, min_value=0.0, max_value=10.0, step=0.1
    )


# Get Data from Flask API

input_data = [
    sepal_length,
    sepal_width,
    petal_length,
    petal_width,
]
# FLASK_URL = "http://127.0.0.1:5000/predict"
FLASK_URL = "https://mlops-project-6h7m.onrender.com/"
REQUEST_TIMEOUT = 30  # seconds

try:
    response = requests.post(
        FLASK_URL, json={"features": input_data}, timeout=REQUEST_TIMEOUT
    )

    if response.status_code == 200:
        result = response.json()
        if "error" in result:
            st.error(f"Flask API Error: {result['error']}")
            st.stop()
        predicted_species = result["label"]
        probabilities = list(result["probabilities"].values())
        class_names = list(result["probabilities"].keys())
    else:
        st.error(
            f"Flask API Error: {response.status_code} failed to get response from Flask API"
        )
        st.stop()

except Exception as e:
    st.error(f"Error: {e} failed to get response from Flask API")
    st.stop()

## Show predicted species and probabilities
st.markdown("## Prediction Result")
st.success(f"Predicted Species: {predicted_species}")

prob_df = pd.DataFrame({"Species": class_names, "Probability": probabilities})

## Plotting bar chart for prediction probabilities against flower species
fig = px.bar(
    prob_df,
    x="Species",
    y="Probability",
    color="Species",
    color_discrete_sequence=[
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
    ],
    title="Prediction Probability by Species",
    text=prob_df["Probability"].apply(lambda x: f"{x:.2f}"),
)

# Designing the Layout for Bar chart
fig.update_layout(
    plot_bgcolor="#1a1a1a",
    paper_bgcolor="#0a0a0a",
    font=dict(color="white"),
    title_font=dict(size=20, color="#ffa500"),
    yaxis_tickformat=".0%",
    yaxis_range=[0, 1],
)

## This is to make bar chart interactive
fig.update_traces(textposition="outside")


## To display the bar chart
st.plotly_chart(fig, use_container_width=True)

st.markdown("****")

st.caption("Your project is now compeleted")
