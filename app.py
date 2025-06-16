import streamlit as st
import pandas as pd
from PIL import Image
import datetime

st.set_page_config(page_title="HAB Detection System", layout="centered")

# HEADER
st.markdown("""
    <h1 style='text-align: center; color: #2E8B57;'>HAB Detection System</h1>
    <p style='text-align: center; font-size:18px;'>Upload an image or a dataset to start detecting harmful algal blooms.</p>
""", unsafe_allow_html=True)

# INPUT SELECTION
input_choice = st.radio("Select Input Method:", ["Image Upload", "CSV File"], horizontal=True)

# IMAGE UPLOAD SECTION
if input_choice == "Image Upload":
    st.markdown("""
        <div style='background-color: #e8f5e9; padding: 20px; border-radius: 10px; box-shadow: 0 4px 10px rgba(34,139,34,0.1);'>
        <h3 style='color:#2E8B57;'> Upload Image for Prediction</h3>
    """, unsafe_allow_html=True)

    uploaded_image = st.file_uploader(" Upload Image *", type=["jpg", "jpeg", "png"], key="image")

    region = st.selectbox(" Region *", ["-- Select Region --", "North", "South", "East", "West", "Midwest"])
    date_selected = st.date_input("Date *", value=datetime.date.today())
    latitude = st.number_input(" Latitude *", format="%.6f")
    longitude = st.number_input(" Longitude *", format="%.6f")

    if uploaded_image is not None:
        st.image(uploaded_image, caption=" Confirm Uploaded Image", use_container_width=True)

    if st.button(" Predict from Image"):
        errors = []
        if uploaded_image is None:
            errors.append(" Image is required.")
        if region == "-- Select Region --":
            errors.append(" Region must be selected.")
        if latitude == 0.0 or longitude == 0.0:
            errors.append(" Valid coordinates required.")

        if errors:
            for err in errors:
                st.markdown(
                    f"<div style='border:1px solid red; padding:10px; margin:5px 0; border-radius:6px; background:#ffe6e6; color:red;'>{err}</div>",
                    unsafe_allow_html=True
                )
        else:
            st.success(" Predicted Severity: High")
            st.info(" Estimated Risk Duration: 5 Days")
            st.markdown(f" **Region:** {region} &nbsp;&nbsp;&nbsp;&nbsp;  **Date:** {date_selected}")
            st.markdown(f" **Lat:** `{latitude}` &nbsp;&nbsp; **Lon:** `{longitude}`")

    st.markdown("</div>", unsafe_allow_html=True)

# CSV UPLOAD SECTION
elif input_choice == "CSV File":
    st.markdown("""
        <div style='background-color: #f1f8e9; padding: 20px; border-radius: 10px; box-shadow: 0 4px 10px rgba(46,139,87,0.1);'>
        <h3 style='color:#2E8B57;'> Upload CSV File</h3>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(" Upload CSV *", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader(" Data Preview")
        st.dataframe(df, use_container_width=True)

        if st.button(" Predict from CSV"):
            df["Predicted_Severity"] = "Moderate"  # Dummy prediction
            df["Estimated_Risk_Days"] = 4

            st.success(" File Upload Complete")
            st.subheader(" Results")
            st.dataframe(df, use_container_width=True)

            csv_out = df.to_csv(index=False).encode("utf-8")
            st.download_button(" Download CSV", data=csv_out, file_name="hab_predictions.csv")

    st.markdown("</div>", unsafe_allow_html=True)