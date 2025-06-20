import streamlit as st 
import datetime 
import random 

st.set_page_config(page_title= "HAB DETECTION SYSTEM",layout="centered")
st.markdown("<h2 style='text-align: center; color: green;'>HAB Detection System</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: cneeter;'>Choose your input values below.</p", unsafe_allow_html=True)
st.markdown("---")

with st.form("hab_form"):
    st.subheader("Fill the Details")
    uid = st.text_input("Unique ID")
    data_provider = st.selectbox("Data Provider", [
        "Indiana State Department of Health",
        "California Environmental Data Exchange Network",
        "NC Division of Water Resources NC Department of Environmental Quality",
        "Bureau of Water Kansas Department of Health and Environment",
        "US Army Corps of Engineers",
        "EPA National Aquatic Research Survey",
        "EPA Central Data Exchange",
        "Other"
    ])
    region = st.selectbox("Region", ["Northeast","West","Midwest","South"])
    distance_to_water = st.number_input("Distance to Water (in meters)", min_value=0.0, step=1.0)
    lat = st.number_input("Latitude", format="%.6f")
    lon = st.number_input("Longitude", format="%.6f")
    date = st.date_input("Sample Collection Date", value=datetime.date.today())
    time = st.time_input("Sample Collection Time", value=datetime.time(12, 0))
    abun = st.number_input("Abundance Level", min_value=0.0, step=0.01)
    submit = st.form_submit_button("Submit")
if submit:
    if not uid or not data_provider or not region or not lat or not lon or not date or not time or not abun:
        st.error("Please fill in all the required fields.")
    else:
        result = random.choice(["Toxic", "Non-Toxic"])
        st.success("Your Form has Submitted Succesfully!")
        st.markdown("---")
        st.markdown(
            f"<h3 style='text-align: center; color: {'red' if result == 'Toxic' else 'green'};'> Prediction:{result}</h3>",
            unsafe_allow_html=True
        )

    