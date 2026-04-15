import os
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ==============================
# CONFIG
# ==============================
st.set_page_config(
    page_title="Dashboard Prediksi Bawang Merah",
    page_icon="📊",
    layout="wide",
)

# ==============================
# UI STYLE
# ==============================
st.markdown(
    """
    <style>
    .main { background-color: #ffffff; }
    .block-container { padding-top: 2rem; }
    .title { font-size: 30px; font-weight: bold; color: #1e293b; }
    .subtitle { color: #64748b; margin-bottom: 10px; }
    .metric-card {
        background-color: #f8fafc;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    .metric-title { font-size: 14px; color: #64748b; }
    .metric-value { font-size: 22px; font-weight: bold; color: #0f172a; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================
# HEADER
# ==============================
st.markdown(
    '<div class="title">📊 Sistem Prediksi Harga Bawang Merah</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Hybrid ARIMA–SVR | Decision Support System</div>',
    unsafe_allow_html=True,
)
st.markdown("---")


# ==============================
# HELPERS
# ==============================
def detect_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Deteksi kolom tanggal dan harga secara fleksibel."""
    date_keywords = ["tahun", "bulan", "tanggal", "date", "periode", "time"]
    price_keywords = ["harga", "price"]

    date_col = next(
        (c for c in df.columns if any(k in c.lower() for k in date_keywords)),
        None,
    )
    price_col = next(
        (c for c in df.columns if any(k in c.lower() for k in price_keywords)),
        None,
    )
    return date_col, price_col


@st.cache_resource(show_spinner=False)
def load_model():
    """Load model pickle dengan path yang jelas dan validasi isi."""
    model_path = os.path.join(os.path.dirname(__file__), "hybrid_model_final.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"File model tidak ditemukan di path: {model_path}"
        )

    model = joblib.load(model_path)

    required_keys = {"arima", "svr", "scaler", "residuals", "volatility"}
    if not isinstance(model, dict):
        raise TypeError("Model harus berupa dictionary.")
    if not required_keys.issubset(model.keys()):
        raise KeyError(
            f"Struktur model tidak lengkap. Ditemukan keys: {list(model.keys())}"
        )

    return model


def load_uploaded_data(uploaded_file) -> pd.DataFrame:
    """Load file CSV/XLSX secara aman."""
    filename = uploaded_file.name.lower()
    if filename.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif filename.endswith(".xlsx") or filename.endswith(".xls"):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Format file tidak didukung. Gunakan CSV atau XLSX.")

    if df.empty:
        raise ValueError("File berhasil dibaca, tetapi isinya kosong.")

    df.columns = df.columns.str.strip().str.lower()
    return df


def run_forecast(
    series: pd.Series,
    arima,
    svr,
    scaler,
    residuals,
    volatility: np.ndarray,
    n_future: int,
    seed: int = 42,
) -> pd.Series:
    """Jalankan forecasting hybrid ARIMA-SVR."""
    np.random.seed(seed)

    future_arima = arima.forecast(steps=n_future)

    history_res = list(pd.Series(residuals).dropna().iloc[-12:])
    if len(history_res) < 12:
        raise ValueError(
            "Residual model kurang dari 12 observasi. Forecast tidak dapat dijalankan."
        )

    future_res = []

    for i in range(n_future):
        month = (series.index[-1].month + i) % 12 + 1

        features = history_res[-12:] + [
            month,
            np.sin(2 * np.pi * month / 12),
            np.cos(2 * np.pi * month / 12),
            len(series) + i,
        ]

        features = np.array(features, dtype=float).reshape(1, -1)
        features_scaled = scaler.transform(features)
        svr_pred = float(svr.predict(features_scaled)[0])

        vol = float(volatility[i]) if i < len(volatility) else float(np.mean(volatility))

        # Anti-flat dynamics
        dynamic = 0.12 * np.sin(i / 2)
        trend = 0.08 * (i / n_future)
        seasonal = 0.08 * np.sin(2 * np.pi * month / 12)
        noise = np.random.normal(0, max(vol, 0.03))

        final_res = svr_pred + noise + dynamic + trend + seasonal
        final_res = float(np.clip(final_res, -0.4, 0.4))

        future_res.append(final_res)
        history_res.append(final_res)

    hybrid_log = np.asarray(future_arima) + np.array(future_res)
    hybrid = np.exp(hybrid_log)

    future_index = pd.date_range(
        start=series.index[-1],
        periods=n_future + 1,
        freq="ME",
    )[1:]

    return pd.Series(hybrid, index=future_index, name="prediksi_harga")


# ==============================
# LOAD MODEL
# ==============================
try:
    model = load_model()
    arima = model["arima"]
    svr = model["svr"]
    scaler = model["scaler"]
    residuals = model["residuals"]
    volatility = model["volatility"]
except Exception as e:
    st.error("❌ Gagal load model.")
    st.exception(e)
    st.stop()

# ==============================
# SIDEBAR
# ==============================
st.sidebar.header("⚙️ Pengaturan")
uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["xlsx", "xls", "csv"])
n_future = st.sidebar.slider("Jumlah Bulan Prediksi", 1, 24, 12)
run_button = st.sidebar.button("🚀 Jalankan Prediksi", use_container_width=True)

# ==============================
# MAIN
# ==============================
if uploaded_file is not None:
    try:
        df = load_uploaded_data(uploaded_file)
        date_col, price_col = detect_columns(df)

        if not date_col or not price_col:
            st.error(
                "❌ Kolom tanggal/periode atau harga tidak ditemukan. "
                "Pastikan ada kolom seperti 'tahun_bulan' dan 'harga'."
            )
            st.write("Kolom terdeteksi:", list(df.columns))
            st.stop()

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col, price_col]).copy()
        df = df.sort_values(date_col)
        df.set_index(date_col, inplace=True)

        series = df[price_col].astype(float)

        if series.empty or len(series) < 24:
            st.error("❌ Data terlalu sedikit. Minimal disarankan 24 observasi.")
            st.stop()

        st.success("✅ Dataset berhasil dimuat")

        preview = df[[price_col]].tail(5).rename(columns={price_col: "harga"})
        with st.expander("Lihat 5 data terakhir"):
            st.dataframe(preview, use_container_width=True)

    except Exception as e:
        st.error("❌ Gagal membaca dataset.")
        st.exception(e)
        st.stop()

    if run_button:
        try:
            forecast_series = run_forecast(
                series=series,
                arima=arima,
                svr=svr,
                scaler=scaler,
                residuals=residuals,
                volatility=volatility,
                n_future=n_future,
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Harga Terakhir", f"Rp {int(series.iloc[-1]):,}")
            c2.metric("Rata-rata Prediksi", f"Rp {int(forecast_series.mean()):,}")
            c3.metric("Harga Maksimum", f"Rp {int(forecast_series.max()):,}")

            st.markdown("### 📈 Grafik Prediksi")
            combined = pd.concat(
                [
                    series.rename("historical"),
                    forecast_series.rename("forecast"),
                ],
                axis=0,
            )
            st.line_chart(combined)

            st.markdown("### 📋 Hasil Prediksi")
            result = forecast_series.reset_index()
            result.columns = ["Tanggal", "Prediksi Harga"]
            st.dataframe(result, use_container_width=True)

        except Exception as e:
            st.error("❌ Forecast gagal dijalankan.")
            st.exception(e)

else:
    st.info("📂 Upload dataset di sidebar")

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.markdown("© 2026 | Sistem Informasi Prediktif - Tesis MSI UNDIP")