import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# --- БАПТАУЛАР ---
MODEL_FILENAME = 'final_model.keras'
# Төмендегі ID-ді 2-ұяшық біткен соң, Google Drive-тан алып қойыңыз!
GDRIVE_FILE_ID = '1LnqMkWRgVRUhOA9pAGqk0j74tUuPmHis' 

# Модельді жүктеу
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILENAME):
        url = f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}'
        gdown.download(url, MODEL_FILENAME, quiet=False)
    return tf.keras.models.load_model(MODEL_FILENAME)

try:
    model = load_model()
    st.success("Модель жүктелді!")
except:
    st.error("Модель ID қате немесе файл жоқ. Google Drive ID тексеріңіз.")
    model = None

# --- ИНТЕРФЕЙС ---
st.title("🌱 Топырақ Құнарлылығын Анықтау (Smart System)")

col1, col2 = st.columns(2)

with col1:
    st.header("1. Деректерді енгізу")
    moisture = st.slider("Ылғалдылық (Су)", 0.0, 1.0, 0.5)
    salinity = st.slider("Тұздылық", 0.0, 1.0, 0.1)
    urban = st.slider("Қала/Ғимарат тығыздығы", 0.0, 1.0, 0.0)
    agri = st.slider("Өсімдік/Егістік тығыздығы", 0.0, 1.0, 0.5)

with col2:
    st.header("2. Суретті жүктеу")
    uploaded_file = st.file_uploader("Спутник суретін таңдаңыз", type=["jpg", "png"])

if uploaded_file and model:
    # Суретті дайындау
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Жүктелген сурет", use_column_width=True)
    
    img_array = np.array(image.resize((224, 224))) / 255.0
    img_input = np.expand_dims(img_array, axis=0)
    
    # Сандарды дайындау
    tab_input = np.array([[moisture, salinity, urban, agri]])
    
    # Болжау
    prediction = model.predict({'image_input': img_input, 'tabular_input': tab_input})
    score = float(prediction[0][0]) # Float-қа айналдыру (Қатені түзетеді)
    
    # Нәтиже
    st.subheader(f"Құнарлылық: {score:.2f}")
    st.progress(score) # Енді бұл жерде қате шықпайды
    
    if score > 0.7:
        st.success("ӨТЕ ҚҰНАРЛЫ ЖЕР! ✅")
    elif score > 0.4:
        st.warning("ОРТАША ҚҰНАРЛЫЛЫҚ ⚠️")
    else:
        st.error("ҚҰНАРЛЫ ЕМЕС / ТҰЗДЫ / ҚАЛА 🛑")
