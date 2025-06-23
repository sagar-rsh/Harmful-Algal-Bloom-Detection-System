import streamlit as st 
from datetime import date
import requests
from streamlit_folium import st_folium
import folium

st.set_page_config(page_title= "HAB Detection System",layout="wide", initial_sidebar_state="collapsed")

st.title("Harmful Algal Bloom (HAB) Detection System")
st.markdown("""
			Enter data manually to get a prediction from the backend API.            
			""")
st.markdown("---")

# Session state to hold prediction results
if 'prediction' not in st.session_state:
    st.session_state.prediction = None

# API Endpoint URL
API_URL = f"{st.secrets.get("API_URL", "http://localhost:5000")}/predict"

# Layout
input_col, map_col = st.columns([1, 1.5], gap='large')

with input_col:
    st.subheader("Prediction Input")

    # default value
    default_lat, default_lon = 27.6133, -82.7391

    # Store map coordinates in state
    if 'map_center' not in st.session_state:
        st.session_state.map_center = [default_lat, default_lon]

    lat = st.session_state.get('lat', default_lat)
    lon = st.session_state.get('lon', default_lon)

    with st.form("hab_form"):
        lat = st.number_input("Latitude", value=lat, format="%.6f")
        lon = st.number_input("Longitude", value=lon, format="%.6f")

        selected_date = st.date_input(
            "Select a date for the start of the 5-day window",
            min_value=date(2015, 1, 1),
            value=date.today(),
            max_value=date.today(),
            help="The model will predict the status for the 5th day after this date"
        )

        submit = st.form_submit_button(label='Get Prediction', use_container_width=True)

with map_col:
    st.subheader("Pick your location on the map")
    # Centering on Gulf of Mexico
    map_object = folium.Map(location=st.session_state.map_center, zoom_start=5)
    folium.Marker([lat, lon], popup="Selected Location", tooltip="Selected Location").add_to(map_object)

    map_data = st_folium(map_object, width=700, height=500)
    if map_data and map_data["last_clicked"]:
        st.session_state.lat = map_data["last_clicked"]["lat"]
        st.session_state.lon = map_data["last_clicked"]["lng"]
        # Rerun after each click
        st.rerun()


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
                
                st.session_state.prediction = response.json()

                prediction = st.session_state.prediction
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