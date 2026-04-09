import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from PIL import Image
import requests
from io import BytesIO

# ================= CONFIG =================
st.set_page_config(page_title="Dry Bean", layout="wide")

# ================= UI =================
st.markdown("""
<style>

/* BACKGROUND */
.stApp {
    background: linear-gradient(135deg, #1e3c72, #2a5298);
}

/* MAIN */
.block-container {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(18px);
    padding: 2rem;
    border-radius: 20px;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: #0f172a;
    color: white;
}

/* HEADER */
.hero {
    font-size: 34px;
    font-weight: 800;
    color: white;
}

/* KPI CARDS */
.kpi {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}

/* BUTTON */
.stButton>button {
    background: linear-gradient(90deg, #00c853, #2e7d32);
    color: white;
    border-radius: 12px;
    height: 3em;
    font-weight: 700;
}

/* TEXT */
h1,h2,h3,p,label { color: white !important; }

</style>
""", unsafe_allow_html=True)

# ================= LOAD =================
@st.cache_resource
def load():
    model = pickle.load(open("model.pkl","rb"))
    scaler = pickle.load(open("scaler.pkl","rb"))
    le = pickle.load(open("label_encoder.pkl","rb"))
    features = pickle.load(open("features.pkl","rb"))
    return model, scaler, le, features

model, scaler, le, feature_cols = load()

# ================= IMAGE =================
def load_img(url):
    try:
        r = requests.get(url)
        return Image.open(BytesIO(r.content))
    except:
        return None

# ================= BEAN INFO =================
bean_info = {
    "SIRA": (
        "https://storage.googleapis.com/kaggle-datasets-images/2048855/3398802/f484410f3cfb3f7f49d1216f57f18850/dataset-card.jpg",
        "Sira beans are medium-sized, elongated legumes commonly cultivated in Mediterranean regions. They exhibit balanced density and moderate eccentricity, making them structurally stable and easy to analyze. Their consistent geometric properties allow machine learning models to distinguish them effectively from other classes. Due to their predictable morphology and uniform distribution of features, Sira beans are widely used in classification datasets for reliable model training and evaluation."
    ),

    "HOROZ": (
        "https://5.imimg.com/data5/JW/PG/MY-55016025/pahadi-organic-munshyari-rajma-2c-500g-1000x1000.png",
        "Horoz beans are large, dense legumes typically grown in mountainous regions where environmental conditions demand structural resilience. They possess a thick outer coat and a strong internal structure, making them highly durable. Their larger size, high area, and perimeter values allow them to stand out clearly in datasets. In machine learning applications, Horoz beans provide strong feature signals, improving classification accuracy and helping models learn distinct patterns efficiently."
    ),

    "SEKER": (
        "https://5.imimg.com/data5/SELLER/Default/2025/3/494462208/ZZ/KO/ZW/20804870/white-small-kidney-beans-himalayan-1000x1000.jpg",
        "Seker beans are small, rounded legumes known for their smooth surfaces and compact internal structure. They exhibit high uniformity in size and shape, making them ideal for machine learning classification tasks. Their low aspect ratio and high compactness create a tightly clustered feature space. This consistency allows models to learn patterns efficiently and achieve high prediction accuracy, making Seker beans a reliable reference class in many datasets."
    ),

    "BARBUNYA": (
        "https://5.imimg.com/data5/SELLER/Default/2024/6/427675461/FL/CC/YF/58913207/rajma-bhutan-1000x1000.jpg",
        "Barbunya beans are reddish, curved legumes with a distinctive outer texture and irregular shape. They are commonly found in Mediterranean regions and are visually unique compared to other bean types. Their variation in geometry and surface features introduces diversity into classification datasets. This diversity helps machine learning models learn more complex patterns, improving robustness and generalization performance across different classes."
    ),

    "CALI": (
        "https://cdn-img.freshdi.com/640x640/files/8ef0621283f2058e886ee036a4a1f6d8.JPG",
        "Cali beans are oval-shaped legumes characterized by smooth edges and consistent morphology. They are widely cultivated and exhibit balanced geometric features, making them suitable for both statistical analysis and machine learning classification. Their uniform size and structure ensure stable feature extraction, allowing models to generalize well. Cali beans are often used in datasets where consistency and reliability are important for accurate predictions."
    ),

    "DERMASON": (
        "https://tiimg.tistatic.com/fp/1/006/933/natural-organic-sugar-white-kidney-bean-664.jpg",
        "Dermason beans are small, white legumes with smooth surfaces and highly uniform geometry. They are widely used in machine learning datasets due to their consistent structural properties and low variability. Their predictable shape and size make them ideal for training classification models with high precision. Dermason beans are often used as benchmark samples, as they help ensure stable model performance and reliable prediction outcomes."
    ),

    "BOMBAY": (
        "https://upload.wikimedia.org/wikipedia/commons/3/3f/Pinto_Beans_Seeds.jpg",
        "Bombay beans are large, robust legumes known for their thick outer coating and durable internal structure. They are commonly cultivated in warm climates and are valued for their resistance to handling damage during storage and transportation. Their larger size and distinct geometric features make them easily identifiable in classification datasets. In machine learning applications, Bombay beans contribute to improved model accuracy by providing strong and well-defined feature patterns."
    )
}

def norm(x): return x.strip().upper()

# ================= SIDEBAR NAV =================
st.sidebar.title("🌱 Dry Bean")
page = st.sidebar.radio("Navigation", ["Prediction", "Analytics", "Bean Info"])

# ================= LOAD DATA =================
df = pd.read_excel("Beans Multiclass Classification.xlsx")
df.columns = df.columns.str.lower()

# ================= HERO =================
st.markdown('<div class="hero">🌱 Dry Bean Classification</div>', unsafe_allow_html=True)
st.write("Smart ML-driven bean classification with actionable insights and visual analytics.")

# ================= KPI =================
c1,c2,c3 = st.columns(3)
c1.markdown('<div class="kpi">📊 Dataset Size<br><b>{}</b></div>'.format(len(df)), unsafe_allow_html=True)
c2.markdown('<div class="kpi">🌾 Classes<br><b>{}</b></div>'.format(df["class"].nunique()), unsafe_allow_html=True)
c3.markdown('<div class="kpi">⚙ Features<br><b>{}</b></div>'.format(len(feature_cols)), unsafe_allow_html=True)

# ================= PREDICTION =================
if page == "Prediction":

    st.subheader("🔮 Make Prediction")

    data = {}
    for col in feature_cols:
        data[col] = st.number_input(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))

    input_df = pd.DataFrame([data])

    if st.button("🚀 Predict"):
        input_df = input_df.reindex(columns=feature_cols)
        scaled = scaler.transform(input_df)

        pred = model.predict(scaled)
        label = norm(le.inverse_transform(pred)[0])

        st.success(f"Prediction: {label}")

        img, desc = bean_info.get(label, list(bean_info.values())[0])

        col1,col2 = st.columns([1,2])
        im = load_img(img)
        if im:
            col1.image(im, use_container_width=True)
        col2.write(desc)

# ================= ANALYTICS =================
elif page == "Analytics":

    st.subheader("📊 Data Insights")

    counts = df["class"].value_counts().reset_index()
    counts.columns = ["Class","Count"]

    fig = px.bar(counts, x="Class", y="Count", color="Class")
    st.plotly_chart(fig, use_container_width=True)

# ================= INFO =================
else:

    st.subheader("📚 Bean Encyclopedia")

    for k,(img,desc) in bean_info.items():
        st.markdown(f"### {k}")
        im = load_img(img)
        if im:
            st.image(im, width=300)
        st.write(desc)
        st.markdown("---")

# ================= FOOTER =================
st.markdown("🚀 Portfolio Project | Built by Heramba Kakati")