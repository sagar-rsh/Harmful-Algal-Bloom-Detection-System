import streamlit as st 
import datetime 
import requests

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
                print(prediction)
                # prediction = {'is_harmful': 1, 'prediction': 'Toxic'}

                predicted_value = prediction.get("predicted_value")
                if predicted_value:
                    st.error(f"**Status:** {prediction['predicted_label'].capitalize()}")
                else:
                    st.success(f"**Status:** {prediction['predicted_label'].capitalize()}")
                
                st.metric(label="Confidence", value=f"{prediction['confidence_scores'][str(predicted_value)]*100}%")
                
                with st.expander("Show Raw API Response"):
                    st.json(prediction)
        
        except requests.exceptions.RequestException as e:
            st.error(f"**API Error:** Could not connect to the backend service. Please ensure the Docker container is running.")
            st.error(f"Details: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
#adding map logic (tab)
with tabs[1]:
    st.subheader("Pick your location on the map")
    from streamlit_folium import st_folium
    import folium
    # Centering on Gulf of Mexico
    default_lat, default_lon = 25.681137, -89.890137
    map_object = folium.Map(location=[default_lat, default_lon], zoom_start=5)
    st.markdown("**Click on the map to select a location.**")
    map_data = st_folium(map_object, width=700, height=500)
    if map_data and map_data["last_clicked"]:
        selected_lat = map_data["last_clicked"]["lat"]
        selected_lon = map_data["last_clicked"]["lng"]
        st.success(f"Selected Location → Latitude: {selected_lat:.6f}, Longitude: {selected_lon:.6f}")