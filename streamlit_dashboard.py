import streamlit as st 
from datetime import date
import requests
from streamlit_folium import st_folium
import folium

st.set_page_config(page_title= "HAB DETECTION SYSTEM",layout="centered")
st.title("Harmful Algal Bloom (HAB) Detection")
st.markdown("""
			Enter data manually to get a prediction from the backend API.            
			""")
st.markdown("---")

tabs = st.tabs(["Enter your details", "Pick your location on map"])

# API Endpoint URL
API_URL = f"{st.secrets.get("API_URL", "http://localhost:5000")}/predict"

with tabs[0]:
    with st.form("hab_form"):
        st.subheader("Fill the Details")
        lat = st.number_input("Latitude", format="%.6f")
        lon = st.number_input("Longitude", format="%.6f")
        selected_date = st.date_input(
            "Select a date for the start of the 5-day window",
            min_value=date(2015, 1, 1),
            value=date.today(),
            max_value=date.today()
        )
        submit = st.form_submit_button("Submit")

#adding map logic (tab)
with tabs[1]:
    st.subheader("Pick your location on the map")
    # Centering on Gulf of Mexico
    default_lat, default_lon = 25.681137, -89.890137
    map_object = folium.Map(location=[default_lat, default_lon], zoom_start=5)
    st.markdown("**Click on the map to select a location.**")
    map_data = st_folium(map_object, width=700, height=500)
    if map_data and map_data["last_clicked"]:
        lat = map_data["last_clicked"]["lat"]
        lon = map_data["last_clicked"]["lng"]
        st.success(f"Selected Location → Latitude: {lat:.6f}, Longitude: {lon:.6f}")

if submit:
    if not lat and not lon:
        st.error("Please fill in all the required fields.")
    else:
        payload = {
            "longitude": lon,
            "latitude": lat,
            "date": selected_date.strftime("%Y-%m-%d")
        }
        try:
            with st.spinner('Asking the model for a prediction...'):
                response = requests.post(API_URL, json=payload)
                response.raise_for_status()  # Raise an exception for bad status codes
                
                prediction = response.json()

                predicted_label = prediction.get("predicted_label")
                if predicted_label == 'toxic':
                    st.error(f"**Status:** {predicted_label.capitalize()}")
                else:
                    st.success(f"**Status:** {predicted_label.capitalize()}")
                
                st.metric(label="Confidence", value=f"{float(prediction['confidence_scores'][predicted_label])*100}%")
                
                with st.expander("Show Raw API Response"):
                    st.json(prediction)
        
        except requests.exceptions.RequestException as e:
            st.error(f"**API Error:** Could not connect to the backend service. Please ensure the Docker container is running.")
            st.error(f"Details: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")