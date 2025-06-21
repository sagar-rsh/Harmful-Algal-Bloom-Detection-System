import streamlit as st 
import datetime
import requests
st.set_page_config(page_title="HAB Detection System", layout="centered")
st.markdown("<h2 style='text-align: center; color: violet;'>HAB Detection System</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Please fill out the details below to get a prediction.</p>", unsafe_allow_html=True)
st.markdown("---")

with st.form("hab_form"):
    st.subheader("Sample Details")

    uid = st.text_input("Unique ID")
    data_provider = st.selectbox("Data Provider", [
        "Indiana State Department of Health",
        "California Environmental Data Exchange Network",
        "NC Division of Water Resources NC Department of Environmental Quality",
        "Bureau of Water Kansas Department of Health and Environment",
        "US Army Corps of Engineers",
        "EPA National Aquatic Research Survey",
        "EPA Central Data Exchange"
        "Other"
    ])
    region = st.selectbox("Region", ["Northeast","West","Midwest","South"])
    distance_to_water_m = st.number_input("Distance to Water (meters)", min_value=0.0, step=1.0)
    lat = st.number_input("Latitude", format="%.6f")
    lon = st.number_input("Longitude", format="%.6f")
    date = st.date_input("Sample Collection Date", value=datetime.date.today())
    time = st.time_input("Sample Collection Time", value=datetime.time(12,0))
    abun = st.number_input("Abundance Level", min_value=0.0, step=0.01)

    submit = st.form_submit_button("Submit")

if submit:
    if not uid or not data_provider or not region:
        st.error("Please complete all required fields.")
    else:
        input_data = {
            "uid": uid,
            "data_provider": data_provider,
            "region": region,
            "distance_to_water_m": distance_to_water_m,
            "lat": lat,
            "lon": lon,
            "date": date.strftime("%T-%m-%d"),
            "time": time.strftime("%H:%M:%S"),
            "abun": abun
        }
        try:
            response = requests.post("http://localhost:5000/predict", json=input_data)
            result = response.json().get("prediction")

            st.success("Your form has been submitted successfully!")
            st.markdown("---")
            st.markdown(
                f"<h3 style='text-align: center; color: {'red' if result == 'Toxic' else 'green'};'>Prediction: {result}</h3>",
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error("Could not reach backend or received invalid response.")
            st.markdown("---")
            st.markdown(
                f"<h3 style='text-align: center; color: green;'>Prediction: Backend error or unreachable</h3>",
                unsafe_allow_html=True
            )