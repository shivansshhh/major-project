import streamlit as st
import requests
import pandas as pd
import os
import pickle
from fpdf import FPDF
from transformers import MarianMTModel, MarianTokenizer
from openai import OpenAI
import random

# === Custom CSS for better look ===
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@500;700&display=swap');
html, body, [class*="css"]  {
    font-family: 'Montserrat', sans-serif !important;
    background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%);
}
h1, h2, h3, h4 {
    color: #2d3a4b;
    font-weight: 700;
}
.stButton>button, .stDownloadButton>button {
    background: linear-gradient(145deg, #6dd5ed 0%, #2193b0 100%);
    color: white;
    border: none;
    border-radius: 12px;
    box-shadow: 0 4px 14px 0 rgba(33,147,176,0.15), 0 2px 2px 0 rgba(0,0,0,0.05);
    padding: 0.75em 2em;
    font-size: 1.1em;
    font-weight: 600;
    transition: all 0.2s ease;
    margin-bottom: 0.5em;
}
.stButton>button:hover, .stDownloadButton>button:hover {
    background: linear-gradient(145deg, #2193b0 0%, #6dd5ed 100%);
    box-shadow: 0 8px 24px 0 rgba(33,147,176,0.25), 0 4px 4px 0 rgba(0,0,0,0.08);
    transform: translateY(-2px) scale(1.04);
}
</style>
""", unsafe_allow_html=True)

# --- Centered Main Heading ---
st.markdown(
    "<h1 style='text-align: center; color: #2d3a4b; font-size: 2.8em;'>NPK PREDICTION</h1>",
    unsafe_allow_html=True
)
st.markdown("---")

# === Initialize OpenAI client ===
client = OpenAI(api_key="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")

# === Define Base Directory for local files ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# === Load ML models for NPK prediction ===
models = {
    "N": pickle.load(open(os.path.join(BASE_DIR, "models", "N_best_model.pkl"), "rb")),
    "P": pickle.load(open(os.path.join(BASE_DIR, "models", "P_best_model.pkl"), "rb")),
    "K": pickle.load(open(os.path.join(BASE_DIR, "models", "K_best_model.pkl"), "rb")),
}

# === MarianMT Setup for Hindi translation ===
model_name = "Helsinki-NLP/opus-mt-en-hi"
tokenizer = MarianTokenizer.from_pretrained(model_name)
translator = MarianMTModel.from_pretrained(model_name)

def translate_text_openai(text, target_language):
    prompt = f"Translate the following text to {target_language}:\n\n{text}"
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful translator."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1500
    )
    return completion.choices[0].message.content

def get_weather_openweathermap(lat, lon):
    api_key = "363a52b65a3c7a3777b25738999c3d5d"
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    response = requests.get(url).json()
    temperature = response["main"]["temp"]
    humidity = response["main"]["humidity"]
    rainfall = response.get("rain", {}).get("1h", 0.0)
    return temperature, humidity, rainfall

def generate_haiku():
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "write a haiku about ai"}]
    )
    return completion.choices[0].message.content

def generate_smart_summary_openai(n, p, k, temp, humidity, rainfall, crop):
    prompt = f"""
You are an expert agricultural advisor.

Given the following data about a crop and its growing conditions, generate a summary and practical advice for a farmer.

Crop: {crop}
Predicted Fertilizer Levels:
- Nitrogen (N): {n:.2f}
- Phosphorus (P): {p:.2f}
- Potassium (K): {k:.2f}

Current weather conditions:
- Temperature: {temp}°C
- Humidity: {humidity}%
- Rainfall: {rainfall} mm

Write your answer in **three separate paragraphs**. 
- The first paragraph should explain what the predicted NPK values mean for the crop's growth.
- The second paragraph should advise what the farmer should do, especially if temperature rises or weather trends affect crop health.
- The third paragraph should explain why it is important for the farmer to know this information.
Separate each paragraph with a blank line.
Make sure each paragraph is at least 4-5 lines long.
"""
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful agricultural advisor."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1350
    )
    summary = completion.choices[0].message.content.strip()
    return summary

def predict_and_summarize(temperature, humidity, ph, rainfall, crop, target_language):
    try:
        features = [[temperature, humidity, ph, rainfall]]
        n_pred = models["N"].predict(features)[0]
        p_pred = models["P"].predict(features)[0]
        k_pred = models["K"].predict(features)[0]

        report = (
            f"Crop: {crop}\n"
            f"Input Parameters:\n  Temperature: {temperature} °C\n  Humidity: {humidity} %\n"
            f"  Soil pH: {ph}\n  Rainfall: {rainfall} mm\n\n"
            f"Predicted Optimal Fertilizer Levels:\n  Nitrogen (N): {n_pred:.2f}\n"
            f"  Phosphorus (P): {p_pred:.2f}\n  Potassium (K): {k_pred:.2f}"
        )

        summary_text = generate_smart_summary_openai(n_pred, p_pred, k_pred, temperature, humidity, rainfall, crop)

        if target_language.lower() != "english":
            summary_text = translate_text_openai(summary_text, target_language)

        return report, summary_text

    except Exception as e:
        st.error(f"Error during prediction or summary generation: {e}")
        return None, None

WRB_SOIL_CLASSES = {
    "Cambisols": "Young soils with beginning horizon development, often fertile.",
    "Chernozems": "Rich black soils found in grassland regions, high in organic matter.",
    "Ferralsols": "Highly weathered soils found in humid tropics, low fertility.",
    "Podzols": "Acidic soils with subsurface accumulation of organic material and iron.",
    "Gleysols": "Waterlogged soils with greyish colors, found in wetlands.",
    "Andosols": "Volcanic soils, highly porous and fertile.",
    "Solonchaks": "Saline soils often found in arid and semi-arid regions.",
    "Vertisols": "Clay-rich soils that expand when wet and crack when dry.",
    "Regosols": "Very young soils with little profile development.",
    "Arenosols": "Sandy soils with low water retention and fertility."
}

def get_random_soil_classes(n=5):
    selected = random.sample(list(WRB_SOIL_CLASSES.items()), n)
    class_names = [cls[0] for cls in selected]
    descriptions = "\n".join([f"• **{cls}**: {desc}" for cls, desc in selected])
    return class_names, descriptions

# --- Session State Defaults ---
if "location_source" not in st.session_state:
    st.session_state.location_source = None
if "loc_lat" not in st.session_state:
    st.session_state.loc_lat = 12.97
if "loc_lon" not in st.session_state:
    st.session_state.loc_lon = 77.59
if "weather_data" not in st.session_state:
    st.session_state.weather_data = {}
if "avg_temp_16" not in st.session_state:
    st.session_state.avg_temp_16 = 0
if "rainfall_16" not in st.session_state:
    st.session_state.rainfall_16 = 0
if "humidity_16" not in st.session_state:
    st.session_state.humidity_16 = 0
if "report" not in st.session_state:
    st.session_state.report = None
if "summary" not in st.session_state:
    st.session_state.summary = None

# --- Sidebar Navigation ---
tab = st.sidebar.radio(
    "Navigation",
    [
        "Location Finder",
        "Weather Forecast",
        "Text Translator",
        "Generate Report & Summary"
    ],
    key="sidebar_tab"
)

# --- Tab 1: Location Finder ---
if tab == "Location Finder":
    st.header("Location Finder")
    st.markdown("Choose your location using one of the following methods:")

    col1, col2 = st.columns(2)
    with col1:
        ip_disabled = st.session_state.location_source not in [None, "ip"]
        ip_location_btn = st.button("📍 Get My Location (IP-based)", disabled=ip_disabled, key="ip_btn_tab2")
        if ip_location_btn:
            try:
                response = requests.get("http://ip-api.com/json")
                data = response.json()
                if data["status"] == "success":
                    st.session_state.loc_lat = data['lat']
                    st.session_state.loc_lon = data['lon']
                    st.session_state.location_source = "ip"
                    st.success(f"Detected: {data['city']}, {data['regionName']} ({data['country']})")
                else:
                    st.error("Unable to fetch location. Status: " + data.get("message", "Unknown error"))
            except Exception as e:
                st.error(f"Error fetching location: {e}")

    with col2:
        dropdown_disabled = st.session_state.location_source not in [None, "dropdown"]
        cities = {
            "Bangalore": (12.9716, 77.5946),
            "Hubli": (15.3647, 75.1240),
            "Mysore": (12.2958, 76.6394),
            "Mangalore": (12.9141, 74.8560),
            "Belgaum": (15.8497, 74.4977),
            "Davangere": (14.4644, 75.9217),
            "Tumkur": (13.3422, 77.1010),
            "Gulbarga": (17.3297, 76.8343),
            "Bellary": (15.1394, 76.9214),
            "Shimoga": (13.9299, 75.5681),
            "Raichur": (16.2076, 77.3463),
            "Bidar": (17.9133, 77.5301),
            "Udupi": (13.3409, 74.7421),
            "Hospet": (15.2695, 76.3871),
            "Gadag": (15.4300, 75.6295),
            "Chitradurga": (14.2306, 76.3980),
            "Kolar": (13.1357, 78.1326),
            "Mandya": (12.5210, 76.8950),
            "Hassan": (13.0074, 76.0960),
            "Chikmagalur": (13.3152, 75.7750),
            "Chennai": (13.0827, 80.2707),
            "Coimbatore": (11.0168, 76.9558),
            "Madurai": (9.9252, 78.1198),
            "Tiruchirappalli": (10.7905, 78.7047),
            "Salem": (11.6643, 78.1460),
            "Erode": (11.3410, 77.7172),
            "Vellore": (12.9165, 79.1325),
            "Tirunelveli": (8.7139, 77.7567),
            "Thoothukudi": (8.7642, 78.1348),
            "Dindigul": (10.3673, 77.9803),
            "Nagercoil": (8.1780, 77.4280),
            "Hosur": (12.7400, 77.8253),
            "Cuddalore": (11.7447, 79.7680),
            "Kanchipuram": (12.8342, 79.7036),
            "Karur": (10.9601, 78.0766),
            "Sivakasi": (9.4491, 77.7984),
            "Thanjavur": (10.7867, 79.1378),
            "Tiruvannamalai": (12.2253, 79.0747),
            "Namakkal": (11.2196, 78.1670),
            "Tiruppur": (11.1085, 77.3411),
            "Thiruvananthapuram": (8.5241, 76.9366),
            "Kochi": (9.9312, 76.2673),
            "Kozhikode": (11.2588, 75.7804),
            "Thrissur": (10.5276, 76.2144),
            "Kollam": (8.8932, 76.6141),
            "Kannur": (11.8745, 75.3704),
            "Alappuzha": (9.4981, 76.3388),
            "Kottayam": (9.5916, 76.5222),
            "Palakkad": (10.7867, 76.6548),
            "Malappuram": (11.0734, 76.0740),
            "Pathanamthitta": (9.2646, 76.7874),
            "Idukki": (9.8500, 77.0000),
            "Wayanad": (11.6854, 76.1316),
            "Kasargod": (12.5000, 74.9900),
            "Delhi": (28.6139, 77.2090),
            "Mumbai": (19.0760, 72.8777),
            "Hyderabad": (17.3850, 78.4867),
            "Kolkata": (22.5726, 88.3639),
            "Pune": (18.5204, 73.8567),
            "Ahmedabad": (23.0225, 72.5714),
            "Jaipur": (26.9124, 75.7873),
            "Lucknow": (26.8467, 80.9462),
            "Bhopal": (23.2599, 77.4126),
            "Patna": (25.5941, 85.1376)
        }
        selected_city = st.selectbox("Select a City", ["None"] + list(cities.keys()), disabled=dropdown_disabled, key="city_tab2")
        if selected_city != "None" and st.session_state.location_source in [None, "dropdown"]:
            lat, lon = cities[selected_city]
            st.session_state.loc_lat = lat
            st.session_state.loc_lon = lon
            st.session_state.location_source = "dropdown"
            st.success(f"Coordinates for {selected_city} set: ({lat}, {lon})")

    st.subheader("Or Enter Location Manually")
    manual_disabled = st.session_state.location_source not in [None, "manual"]
    lat_manual = st.number_input("Latitude", value=st.session_state.get("loc_lat", 12.97), key="manual_lat_tab2", disabled=manual_disabled)
    lon_manual = st.number_input("Longitude", value=st.session_state.get("loc_lon", 77.59), key="manual_lon_tab2", disabled=manual_disabled)
    if not manual_disabled:
        st.session_state.loc_lat = lat_manual
        st.session_state.loc_lon = lon_manual
        st.session_state.location_source = "manual"

# --- Tab 2: Weather Forecast ---
elif tab == "Weather Forecast":
    st.header("Weather Forecast")
    lat = st.session_state.get("loc_lat", 12.97)
    lon = st.session_state.get("loc_lon", 77.59)
    if st.button("Get Forecast for Entered Location", key="forecast_btn_tab3"):
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,"
            f"windspeed_10m_max,windgusts_10m_max,uv_index_max,sunshine_duration,"
            f"relative_humidity_2m_max,relative_humidity_2m_min"
            f"&timezone=Asia%2FKolkata&forecast_days=16"
        )
        response = requests.get(url)
        data = response.json()
        temp_max_16 = data["daily"]["temperature_2m_max"][-2]
        temp_min_16 = data["daily"]["temperature_2m_min"][-2]
        rainfall_16 = data["daily"]["precipitation_sum"][-2]
        humidity_max_16 = data["daily"]["relative_humidity_2m_max"][-2]
        humidity_min_16 = data["daily"]["relative_humidity_2m_min"][-2]
        windspeed_16 = data["daily"]["windspeed_10m_max"][-2]
        windgusts_16 = data["daily"]["windgusts_10m_max"][-2]
        humidity_16 = (humidity_max_16 + humidity_min_16) / 2
        avg_temp_16 = (temp_max_16 + temp_min_16) / 2
        st.session_state['avg_temp_16'] = avg_temp_16
        st.session_state['rainfall_16'] = rainfall_16
        st.session_state['humidity_16'] = humidity_16

        df = pd.DataFrame(data['daily'])
        df['time'] = pd.to_datetime(df['time'])

        st.subheader("📅 Daily Forecast Data")
        st.dataframe(df)

        st.subheader("🌡️ Temperature Forecast")
        st.line_chart(df.set_index('time')[['temperature_2m_min', 'temperature_2m_max']])

        st.subheader("🌧️ Precipitation Forecast")
        st.bar_chart(df.set_index('time')[['precipitation_sum']])

        st.subheader("🌬️ Wind Speed & Gusts")
        st.line_chart(df.set_index('time')[['windspeed_10m_max', 'windgusts_10m_max']])

        st.subheader("💧 Humidity Levels")
        st.line_chart(df.set_index('time')[['relative_humidity_2m_min', 'relative_humidity_2m_max']])

        st.subheader("🌞 Sun & UV Index")
        st.line_chart(df.set_index('time')[['sunshine_duration', 'uv_index_max']])

# --- Tab 3: Text Translator ---
elif tab == "Text Translator":
    st.header("Text Translator")
    user_input = st.text_area("Enter text to translate (leave empty to generate a haiku):", key="text_translator_tab4")
    target_language = st.selectbox("Select Language", ["Hindi", "Kannada", "Telugu"], key="lang_tab4")
    if st.button("Generate / Translate Text", key="translate_btn_tab4"):
        original_text = user_input.strip() if user_input else generate_haiku()
        translated_text = translate_text_openai(original_text, target_language)
        st.subheader("Original Text:")
        st.write(original_text)
        st.subheader(f"Translated to {target_language}:")
        st.write(translated_text)
        file_content = f"Original Text:\n{original_text}\n\nTranslated to {target_language}:\n{translated_text}"
        st.download_button(
            label="📥 Download Translation",
            data=file_content.encode("utf-8"),
            file_name="translated_text.txt",
            mime="text/plain"
        )

# --- Tab 4: Generate Report & Summary (NPK Prediction) ---
elif tab == "Generate Report & Summary":
    st.header("NPK PREDICTION")
    st.markdown("Fill the form below and generate your NPK report.")

    input_mode = st.radio("Choose Input Mode:", ["Manual", "Satellite-based (Auto)"], key="input_mode_tab1_report")

    if input_mode == "Manual":
        st.subheader("🧪 Enter Soil & Weather Parameters Manually")
        temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0, key="temp_manual_report")
        humidity = st.number_input("Humidity (%)", 0.0, 100.0, 70.0, key="humidity_manual_report")
        ph = st.number_input("Soil pH", 3.0, 9.0, 6.5, key="ph_manual_report")
        rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 100.0, key="rainfall_manual_report")
    else:
        st.subheader("📡 Satellite-based Weather Data")
        lat = st.session_state.get("loc_lat", 0.0)
        lon = st.session_state.get("loc_lon", 0.0)
        st.write(f"📍 Using Coordinates: Latitude = `{lat}`, Longitude = `{lon}`")
        ph = st.number_input("Soil pH", 3.0, 9.0, 6.5, key="ph_sat_report")
        if st.button("Fetch Weather Data", key="fetch_weather_tab1_report"):
            try:
                temperature, humidity, rainfall = get_weather_openweathermap(lat, lon)
                st.session_state.weather_data = {
                    "temperature": temperature,
                    "humidity": humidity,
                    "rainfall": rainfall
                }
                st.success(f"Fetched weather: Temp={temperature}°C, Humidity={humidity}%, Rainfall={rainfall}mm")
            except Exception as e:
                st.error(f"Error fetching weather: {e}")
        if st.session_state.weather_data:
            temperature = st.session_state.weather_data.get("temperature", 0)
            humidity = st.session_state.weather_data.get("humidity", 0)
            rainfall = st.session_state.weather_data.get("rainfall", 0)
            st.write(f"Using weather data: Temp={temperature}°C, Humidity={humidity}%, Rainfall={rainfall}mm")
        else:
            temperature = humidity = rainfall = 0

    crop = st.selectbox("🌿 Select a crop:", [
        "rice", "wheat", "maize", "cotton", "sugarcane",
        "barley", "millet", "sorghum", "groundnut", "soybean",
        "sunflower", "potato", "tomato", "onion", "carrot"
    ], key="crop_tab1_report")

    target_language = st.selectbox("Select Report Language", ["English", "Hindi", "Kannada", "Telugu"], key="lang_tab1_report")

    if st.button("Predict NPK & Generate Report", key="predict_npk_tab1_report"):
        soil_classes, soil_desc_english = get_random_soil_classes(n=5)
        soil_desc_translated = translate_text_openai(soil_desc_english, target_language)
        if soil_classes:
            soil_info_text = (
                "🧪 **Soil Classification:**\n"
                + ", ".join([f"**{cls}**" for cls in soil_classes]) + "\n\n"
                f"📘 **Descriptions (in {target_language}):**\n{soil_desc_translated}"
            )
        else:
            soil_info_text = "⚠️ No soil classification data available."

        avg_temp_16 = st.session_state.get('avg_temp_16', 0)
        rainfall_16 = st.session_state.get('rainfall_16', 0)
        humidity_16 = st.session_state.get('humidity_16', 0)
        forecast_section = (
            "📅 **15th Day Weather Forecast:**\n"
            f"- Crop: {crop}\n"
            f"- Avg Temperature: {avg_temp_16:.1f} °C\n"
            f"- Humidity: {humidity_16:.1f} %\n"
            f"- Rainfall: {rainfall_16:.1f} mm\n"
            f"- Soil pH: {ph}\n"
        )

        report, summary = predict_and_summarize(temperature, humidity, ph, rainfall, crop, target_language)
        if report and summary:
            report += f"\n\n{forecast_section}"
            report += f"\n\n{soil_info_text}"
            st.session_state.report = report
            st.session_state.summary = summary
            st.success("NPK Report & Summary generated! Scroll down to see your report.")

    # Show the report and summary after generation
    if st.session_state.report and st.session_state.summary:
        st.subheader("Fertilizer Report")
        st.code(st.session_state.report)
        st.subheader("Summary & Recommendations")
        st.write(st.session_state.summary)
        full_text = f"Fertilizer Report:\n{st.session_state.report}\n\nSummary & Recommendations:\n{st.session_state.summary}"
        st.download_button(
            label="📥 Download Report & Summary (TXT)",
            data=full_text.encode("utf-8"),
            file_name="npk_report_summary.txt",
            mime="text/plain"
        )
    else:
        st.info("No report generated yet. Please fill the form and click 'Predict NPK & Generate Report' above.")

