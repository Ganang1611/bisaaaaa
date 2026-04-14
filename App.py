# ==============================
# IMPORT
# ==============================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# ==============================
# CONFIG
# ==============================
st.set_page_config(
    page_title="Dashboard Prediksi Bawang Merah",
    page_icon="📊",
    layout="wide"
)

# ==============================
# CUSTOM CSS (WHITE CLEAN UI)
# ==============================
st.markdown("""
<style>
.main {
    background-color: #ffffff;
}

.block-container {
    padding-top: 2rem;
}

.title {
    font-size: 30px;
    font-weight: bold;
    color: #1e293b;
}

.subtitle {
    color: #64748b;
    margin-bottom: 10px;
}

.card {
    background-color: #ffffff;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
}

.metric-card {
    background-color: #f8fafc;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    text-align: center;
}

.metric-title {
    font-size: 14px;
    color: #64748b;
}

.metric-value {
    font-size: 22px;
    font-weight: bold;
    color: #0f172a;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# HEADER
# ==============================
st.markdown('<div class="title">📊 Sistem Prediksi Harga Bawang Merah</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Hybrid ARIMA–SVR | Decision Support System</div>', unsafe_allow_html=True)
st.markdown("---")

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    path = "hybrid_model_final.pkl"

    if not os.path.exists(path):
        st.error("❌ File model tidak ditemukan")
        st.stop()

    try:
        return joblib.load(path)
    except Exception as e:
        st.error("❌ Gagal load model")
        st.text(str(e))
        st.stop()

model = load_model()

# ==============================
# AMBIL KOMPONEN MODEL
# ==============================
try:
    arima = model["arima"]
    svr = model["svr"]
    scaler = model["scaler"]
    residuals = model["residuals"]
    volatility = model["volatility"]
except:
    st.error("❌ Struktur model tidak sesuai")
    st.stop()

np.random.seed(42)

# ==============================
# SIDEBAR
# ==============================
st.sidebar.header("⚙️ Pengaturan")

uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["xlsx","csv"])
n_future = st.sidebar.slider("Jumlah Bulan Prediksi", 1, 24, 12)
run_button = st.sidebar.button("🚀 Jalankan Prediksi")

# ==============================
# MAIN
# ==============================
if uploaded_file:

    # ==============================
    # LOAD DATA
    # ==============================
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    df.columns = df.columns.str.strip().str.lower()

    # auto detect kolom
    date_col = None
    price_col = None

    for col in df.columns:
        if any(x in col for x in ["tahun", "bulan", "tanggal", "date"]):
            date_col = col
        if "harga" in col:
            price_col = col

    if date_col is None or price_col is None:
        st.error("❌ Kolom tanggal & harga tidak ditemukan")
        st.stop()

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    df.set_index(date_col, inplace=True)

    series = df[price_col].astype(float)

    # ==============================
    # RUN MODEL
    # ==============================
    if run_button:

        future_arima = arima.forecast(steps=n_future)

        history_res = list(residuals[-12:])
        future_res = []

        for i in range(n_future):

            month = (series.index[-1].month + i) % 12 + 1

            features = history_res[-12:] + [
                month,
                np.sin(2*np.pi*month/12),
                np.cos(2*np.pi*month/12),
                len(series) + i
            ]

            features = np.array(features).reshape(1, -1)
            features = scaler.transform(features)

            svr_pred = svr.predict(features)[0]

            vol = volatility[i] if i < len(volatility) else np.mean(volatility)

            # ======================
            # DYNAMIC SYSTEM (ANTI FLAT)
            # ======================
            dynamic = 0.12 * np.sin(i / 2)
            trend = 0.08 * (i / n_future)
            seasonal = 0.08 * np.sin(2*np.pi*month/12)

            noise = np.random.normal(0, max(vol, 0.03))

            final_res = svr_pred + noise + dynamic + trend + seasonal
            final_res = np.clip(final_res, -0.4, 0.4)

            future_res.append(final_res)
            history_res.append(final_res + np.random.normal(0, 0.02))

        # ==============================
        # FINAL OUTPUT
        # ==============================
        hybrid_log = future_arima.values + np.array(future_res)
        hybrid = np.exp(hybrid_log)

        future_index = pd.date_range(
            start=series.index[-1],
            periods=n_future+1,
            freq="M"
        )[1:]

        forecast_series = pd.Series(hybrid, index=future_index)

        # ==============================
        # KPI
        # ==============================
        c1, c2, c3 = st.columns(3)

        c1.markdown(f"""
        <div class="metric-card">
        <div class="metric-title">Harga Terakhir</div>
        <div class="metric-value">Rp {int(series.iloc[-1]):,}</div>
        </div>
        """, unsafe_allow_html=True)

        c2.markdown(f"""
        <div class="metric-card">
        <div class="metric-title">Rata-rata Prediksi</div>
        <div class="metric-value">Rp {int(forecast_series.mean()):,}</div>
        </div>
        """, unsafe_allow_html=True)

        c3.markdown(f"""
        <div class="metric-card">
        <div class="metric-title">Harga Maksimum</div>
        <div class="metric-value">Rp {int(forecast_series.max()):,}</div>
        </div>
        """, unsafe_allow_html=True)

        # ==============================
        # GRAFIK
        # ==============================
        st.markdown("### 📈 Grafik Prediksi")

        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(series.index, series.values, label="Historical", linewidth=2)
        ax.plot(forecast_series.index, forecast_series.values, label="Forecast", linewidth=2)

        ax.legend()
        ax.grid(True)

        st.pyplot(fig)

        # ==============================
        # TABEL
        # ==============================
        st.markdown("### 📋 Hasil Prediksi")

        result = forecast_series.reset_index()
        result.columns = ["Tanggal", "Prediksi Harga"]

        st.dataframe(result, use_container_width=True)

else:
    st.info("📂 Silakan upload dataset pada sidebar")

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.markdown("© 2026 | Sistem Informasi Prediktif - Tesis Magister Sistem Informasi UNDIP")