import streamlit as st 
import datetime 
import requests

st.set_page_config(page_title= "HAB DETECTION SYSTEM",layout="centered")
st.title("Harmful Algal Bloom (HAB) Detection")
st.markdown("""
			Enter data manually to get a prediction from the backend API.            
			""")
st.markdown("---")

# API Endpoint URL
API_URL = "http://localhost:5000/predict"

with st.form("hab_form"):
    st.subheader("Fill the Details")
    region = st.selectbox("Region", ["Northeast","West","Midwest","South"], format_func=lambda x: x.lower())
    distance_to_water = st.number_input("Distance to Water (in meters)", min_value=0.0, step=1.0)
    lat = st.number_input("Latitude", format="%.6f")
    lon = st.number_input("Longitude", format="%.6f")

    submit = st.form_submit_button("Submit")

    payload = {
        "region": region,
        "longitude": lon,
        "latitude": lat,
        "distance_to_water_m": distance_to_water
    }
if submit:
    if not region and not lat and not lon:
        st.error("Please fill in all the required fields.")
    else:
        try:
            with st.spinner('Asking the model for a prediction...'):
                response = requests.post(API_URL, json=payload)
                response.raise_for_status()  # Raise an exception for bad status codes
                
                prediction = response.json()
                # prediction = {'is_harmful': 1, 'prediction': 'Toxic'}

                if prediction.get("is_harmful"):
                    st.error(f"**Status:** {prediction['prediction']}")
                else:
                    st.success(f"**Status:** {prediction['prediction']}")
                
                # st.metric(label="Confidence", value=f"{prediction['confidence']:.2%}")
                
                with st.expander("Show Raw API Response"):
                    st.json(prediction)
        
        except requests.exceptions.RequestException as e:
            st.error(f"**API Error:** Could not connect to the backend service. Please ensure the Docker container is running.")
            st.error(f"Details: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")