# main.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import base64
from io import BytesIO

# === Konfigurasi Tema Kreatif dan Elegan ===
st.set_page_config(
    page_title="Xila Studio - Sistem Prediksi Stok Hoodie", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk tema kreatif dan elegan
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Poppins:wght@300;400;500;600;700&display=swap');
    
    :root {
        --primary-color: #6C5CE7;
        --secondary-color: #A29BFE;
        --accent-color: #FECA57;
        --dark-color: #2C3A47;
        --light-color: #FFFFFF;
        --neutral-color: #F8F9FA;
        --text-color: #2C3A47;
        --sidebar-color: #4A69BD;
        --sidebar-text-color: #FFFFFF;
        --card-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        --hover-shadow: 0 15px 35px rgba(0, 0, 0, 0.15);
        --gradient-1: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --gradient-2: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --gradient-3: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --gradient-4: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
    }
    
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    body {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(120deg, #f6d365 0%, #fda085 100%);
        min-height: 100vh;
    }
    
    .stApp {
        background: linear-gradient(120deg, #f6d365 0%, #fda085 100%);
        overflow-x: hidden;
    }
    
    /* Background pattern */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%239C92AC' fill-opacity='0.05'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        opacity: 0.5;
        z-index: -1;
    }
    
    /* Animasi yang lebih menarik */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translate3d(0, 30px, 0);
        }
        to {
            opacity: 1;
            transform: translate3d(0, 0, 0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translate3d(-100px, 0, 0);
        }
        to {
            opacity: 1;
            transform: translate3d(0, 0, 0);
        }
    }
    
    @keyframes float {
        0%, 100% {
            transform: translateY(0);
        }
        50% {
            transform: translateY(-10px);
        }
    }
    
    @keyframes pulse {
        0% {
            transform: scale(1);
            box-shadow: 0 0 0 0 rgba(108, 92, 231, 0.7);
        }
        70% {
            transform: scale(1.1);
            box-shadow: 0 0 0 10px rgba(108, 92, 231, 0);
        }
        100% {
            transform: scale(1);
            box-shadow: 0 0 0 0 rgba(108, 92, 231, 0);
        }
    }
    
    @keyframes shimmer {
        0% {
            background-position: -1000px 0;
        }
        100% {
            background-position: 1000px 0;
        }
    }
    
    .fade-in-up {
        animation: fadeInUp 0.8s ease-out;
    }
    
    .slide-in-left {
        animation: slideInLeft 0.8s ease-out;
    }
    
    .float {
        animation: float 3s ease-in-out infinite;
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    /* Header yang menawan */
    .hero-header {
        background: var(--gradient-1);
        border-radius: 20px;
        padding: 40px;
        margin: 20px 0;
        box-shadow: var(--card-shadow);
        position: relative;
        overflow: hidden;
    }
    
    .hero-header::before {
        content: "";
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 70%);
        animation: shimmer 3s infinite;
    }
    
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 3.5rem;
        font-weight: 700;
        color: white;
        text-align: center;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
    }
    
    .hero-subtitle {
        font-family: 'Poppins', sans-serif;
        font-size: 1.5rem;
        font-weight: 300;
        color: rgba(255,255,255,0.9);
        text-align: center;
        position: relative;
        z-index: 1;
    }
    
    .logo-container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 30px;
        margin-bottom: 20px;
        position: relative;
        z-index: 1;
    }
    
    .logo-container img {
        width: 100px;
        height: 100px;
        border-radius: 50%;
        object-fit: cover;
        border: 4px solid rgba(255,255,255,0.3);
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    .logo-container img:hover {
        transform: scale(1.1);
        box-shadow: 0 15px 40px rgba(0,0,0,0.3);
    }
    
    /* Card yang elegan */
    .elegant-card {
        background: white;
        border-radius: 20px;
        padding: 30px;
        box-shadow: var(--card-shadow);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        margin-bottom: 30px;
    }
    
    .elegant-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 5px;
        background: var(--gradient-2);
    }
    
    .elegant-card:hover {
        transform: translateY(-5px);
        box-shadow: var(--hover-shadow);
    }
    
    /* Metric card yang menawan */
    .metric-card {
        background: white;
        border-radius: 20px;
        padding: 25px;
        box-shadow: var(--card-shadow);
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        border: none;
    }
    
    .metric-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 5px;
        background: var(--gradient-3);
    }
    
    .metric-card:hover {
        transform: translateY(-8px);
        box-shadow: var(--hover-shadow);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--primary-color);
        margin: 10px 0;
        font-family: 'Playfair Display', serif;
    }
    
    .metric-label {
        font-size: 1rem;
        color: var(--text-color);
        font-weight: 500;
    }
    
    /* Sidebar yang elegan */
    .stSidebar {
        background: var(--gradient-1);
        padding: 30px 20px;
    }
    
    .stSidebar .css-1d391kg {
        padding: 20px;
        border-radius: 15px;
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
    }
    
    .stSidebar h1, .stSidebar h2, .stSidebar h3 {
        color: white;
        font-family: 'Playfair Display', serif;
    }
    
    .stSidebar .stSelectbox > div > div > div {
        color: var(--text-color);
        font-weight: 500;
    }
    
    /* Button yang menarik */
    .elegant-button {
        background: var(--gradient-2);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 15px 40px;
        font-weight: 600;
        font-size: 1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
        position: relative;
        overflow: hidden;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .elegant-button::before {
        content: "";
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.5s;
    }
    
    .elegant-button:hover::before {
        left: 100%;
    }
    
    .elegant-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.3);
    }
    
    /* Dataframe yang elegan */
    .elegant-dataframe {
        background: white;
        border-radius: 15px;
        overflow: hidden;
        box-shadow: var(--card-shadow);
    }
    
    /* Grafik yang menawan */
    .chart-container {
        background: white;
        border-radius: 20px;
        padding: 30px;
        box-shadow: var(--card-shadow);
        position: relative;
        overflow: hidden;
    }
    
    .chart-container::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 5px;
        background: var(--gradient-4);
    }
    
    /* Loading animation yang menarik */
    .loading-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 40px;
    }
    
    .loading-spinner {
        width: 50px;
        height: 50px;
        border: 5px solid rgba(255,255,255,0.3);
        border-top-color: var(--accent-color);
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Success animation */
    .success-animation {
        display: inline-block;
        width: 30px;
        height: 30px;
        border-radius: 50%;
        background: var(--gradient-3);
        position: relative;
        animation: success-pop 0.6s ease-out;
    }
    
    @keyframes success-pop {
        0% { transform: scale(0); opacity: 0; }
        50% { transform: scale(1.2); opacity: 1; }
        100% { transform: scale(1); opacity: 1; }
    }
    
    .success-animation::after {
        content: "✓";
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        color: white;
        font-size: 18px;
        font-weight: bold;
    }
    
    /* Decorative elements */
    .decorative-circle {
        position: fixed;
        border-radius: 50%;
        background: var(--gradient-2);
        opacity: 0.1;
        z-index: -1;
    }
    
    .circle-1 {
        width: 300px;
        height: 300px;
        top: 10%;
        right: 5%;
    }
    
    .circle-2 {
        width: 200px;
        height: 200px;
        bottom: 15%;
        left: 5%;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.5rem;
        }
        
        .hero-subtitle {
            font-size: 1.2rem;
        }
        
        .logo-container img {
            width: 80px;
            height: 80px;
        }
    }
</style>
""", unsafe_allow_html=True)

# Elemen dekoratif
st.markdown("""
<div class="decorative-circle circle-1"></div>
<div class="decorative-circle circle-2"></div>
""", unsafe_allow_html=True)

MODEL_PATH = "stok_forecast_model.joblib"
DATA_PATH = "dataset.csv"

# === Fungsi: Baca & Bersihkan Data ===
def load_and_clean_data(filepath):
    try:
        with open(filepath, "r") as f:
            lines = f.readlines()
        
        data_rows = []
        for line in lines:
            stripped = line.strip()
            if stripped and stripped[0].isdigit() and "/" in stripped.split(";")[0]:
                parts = stripped.split(";")
                if len(parts) == 10:
                    data_rows.append(parts)
        
        if not data_rows:
            st.error("Tidak ada data yang valid ditemukan dalam file.")
            return None
        
        df = pd.DataFrame(data_rows, columns=[
            "Tanggal_Transaksi",
            "Nilai_Barang_Dagangan_Bruto(Rp)",
            "Detail_Pendapatan_Bruto_Dengan_Subsidi_Produk_Platform",
            "Produk_Terjual",
            "Pembeli",
            "Tayangan_Halaman",
            "Kunjungan_Halaman_Toko",
            "Pesanan_SKU",
            "Pesanan",
            "Persentase_Konversi"
        ])
        
        df["Tanggal_Transaksi"] = pd.to_datetime(df["Tanggal_Transaksi"], format="%d/%m/%Y")
        for col in ["Produk_Terjual", "Pembeli", "Tayangan_Halaman", "Kunjungan_Halaman_Toko", "Pesanan"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        df = df.dropna(subset=["Tanggal_Transaksi", "Produk_Terjual"]).reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat data: {str(e)}")
        return None

# === Fungsi: Latih & Simpan Model ===
def train_and_save_model(df):
    df_model = df.copy()
    df_model["hari_dalam_minggu"] = df_model["Tanggal_Transaksi"].dt.dayofweek
    df_model["bulan"] = df_model["Tanggal_Transaksi"].dt.month
    df_model["tren_hari"] = np.arange(len(df_model))
    
    X = df_model[[
        "tren_hari", "hari_dalam_minggu", "bulan",
        "Pembeli", "Tayangan_Halaman", "Kunjungan_Halaman_Toko", "Pesanan"
    ]].fillna(df_model.median())
    y = df_model["Produk_Terjual"]
    
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return model, df_model

# === Fungsi: Prediksi Stok Harian ===
def predict_stock_daily(model, df_model, n_hari=30, safety_pct=20):
    last_tren = len(df_model)
    last_date = df_model["Tanggal_Transaksi"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_hari, freq="D")
    
    future_df = pd.DataFrame({
        "Tanggal": future_dates,
        "hari_dalam_minggu": future_dates.dayofweek,
        "bulan": future_dates.month,
        "tren_hari": np.arange(last_tren, last_tren + n_hari),
        "Pembeli": [df_model["Pembeli"].mean()] * n_hari,
        "Tayangan_Halaman": [df_model["Tayangan_Halaman"].mean()] * n_hari,
        "Kunjungan_Halaman_Toko": [df_model["Kunjungan_Halaman_Toko"].mean()] * n_hari,
        "Pesanan": [df_model["Pesanan"].mean()] * n_hari,
    })
    
    X_future = future_df[[
        "tren_hari", "hari_dalam_minggu", "bulan",
        "Pembeli", "Tayangan_Halaman", "Kunjungan_Halaman_Toko", "Pesanan"
    ]]
    
    prediksi = model.predict(X_future)
    prediksi = np.clip(prediksi, 0, None)
    rata2 = prediksi.mean()
    total_stok = rata2 * n_hari + rata2 * (safety_pct / 100)
    
    hasil = pd.DataFrame({
        "Tanggal": future_dates,
        "Prediksi_Produk_Terjual": prediksi
    })
    return hasil, total_stok

# === Fungsi: Prediksi Stok Bulanan ===
def predict_stock_monthly(model, df_model, n_bulan=3, safety_pct=20):
    last_tren = len(df_model)
    last_date = df_model["Tanggal_Transaksi"].max()
    
    # Generate future months
    future_months = []
    current_date = last_date + pd.DateOffset(months=1)
    for i in range(n_bulan):
        month_start = current_date.replace(day=1)
        month_end = (month_start + pd.DateOffset(months=1)) - pd.DateOffset(days=1)
        future_months.append((month_start, month_end))
        current_date = month_end + pd.DateOffset(days=1)
    
    # Create dataframe for monthly prediction
    future_df = pd.DataFrame({
        "Month_Start": [month[0] for month in future_months],
        "Month_End": [month[1] for month in future_months],
        "bulan": [month[0].month for month in future_months],
        "tren_hari": [last_tren + i*30 for i in range(n_bulan)],
        "Pembeli": [df_model["Pembeli"].mean()] * n_bulan,
        "Tayangan_Halaman": [df_model["Tayangan_Halaman"].mean()] * n_bulan,
        "Kunjungan_Halaman_Toko": [df_model["Kunjungan_Halaman_Toko"].mean()] * n_bulan,
        "Pesanan": [df_model["Pesanan"].mean()] * n_bulan,
    })
    
    # Calculate days in each month
    future_df["Days_in_Month"] = future_df.apply(
        lambda row: (row["Month_End"] - row["Month_Start"]).days + 1, axis=1
    )
    
    X_future = future_df[[
        "tren_hari", "bulan",
        "Pembeli", "Tayangan_Halaman", "Kunjungan_Halaman_Toko", "Pesanan"
    ]]
    
    # Predict daily sales and multiply by days in month
    daily_pred = model.predict(X_future)
    monthly_pred = daily_pred * future_df["Days_in_Month"].values
    
    total_stok = monthly_pred.sum() + (monthly_pred.sum() * (safety_pct / 100))
    
    hasil = pd.DataFrame({
        "Bulan": future_df["Month_Start"].dt.strftime('%Y-%m'),
        "Prediksi_Produk_Terjual": monthly_pred,
        "Jumlah_Hari": future_df["Days_in_Month"]
    })
    
    return hasil, total_stok

# === Fungsi: Visualisasi Data Harian ===
def plot_sales_data_daily(df, hasil_prediksi=None):
    plt.figure(figsize=(14, 7))
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Plot data historis dengan gradien warna
    plt.plot(df['Tanggal_Transaksi'], df['Produk_Terjual'], 
             label='Data Historis', color='#6C5CE7', linewidth=3, 
             marker='o', markersize=6, markerfacecolor='#6C5CE7')
    
    # Plot prediksi jika ada
    if hasil_prediksi is not None:
        plt.plot(hasil_prediksi['Tanggal'], hasil_prediksi['Prediksi_Produk_Terjual'], 
                 label='Prediksi', color='#FECA57', linewidth=3, 
                 linestyle='--', marker='s', markersize=6, markerfacecolor='#FECA57')
    
    # Styling yang menawan
    plt.title('📈 Grafik Penjualan Harian', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('📅 Tanggal', fontsize=14, labelpad=10)
    plt.ylabel('🛍️ Jumlah Terjual', fontsize=14, labelpad=10)
    
    # Grid yang elegan
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Legend yang menarik
    plt.legend(facecolor='white', framealpha=0.9, edgecolor='#6C5CE7', 
               fontsize=12, loc='upper left')
    
    # Background yang menarik
    ax = plt.gca()
    ax.set_facecolor('#FAFAFA')
    plt.gcf().set_facecolor('#FAFAFA')
    
    # Styling spine
    for spine in ax.spines.values():
        spine.set_edgecolor('#E0E0E0')
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    return plt

# === Fungsi: Visualisasi Data Bulanan ===
def plot_sales_data_monthly(df, hasil_prediksi=None):
    plt.figure(figsize=(14, 7))
    
    # Prepare monthly historical data
    df_bulanan = df.copy()
    df_bulanan['Bulan'] = df_bulanan['Tanggal_Transaksi'].dt.to_period('M')
    df_bulanan = df_bulanan.groupby('Bulan')['Produk_Terjual'].sum().reset_index()
    df_bulanan['Bulan'] = df_bulanan['Bulan'].astype(str)
    
    # Plot data historis
    bars1 = plt.bar(df_bulanan['Bulan'], df_bulanan['Produk_Terjual'], 
                     label='Data Historis', color='#6C5CE7', alpha=0.8, width=0.6)
    
    # Plot prediksi jika ada
    if hasil_prediksi is not None:
        bars2 = plt.bar(hasil_prediksi['Bulan'], hasil_prediksi['Prediksi_Produk_Terjual'], 
                         label='Prediksi', color='#FECA57', alpha=0.8, width=0.6)
    
    # Styling yang menawan
    plt.title('📊 Grafik Penjualan Bulanan', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('📅 Bulan', fontsize=14, labelpad=10)
    plt.ylabel('🛍️ Jumlah Terjual', fontsize=14, labelpad=10)
    
    # Grid yang elegan
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Legend yang menarik
    plt.legend(facecolor='white', framealpha=0.9, edgecolor='#6C5CE7', 
               fontsize=12, loc='upper left')
    
    # Background yang menarik
    ax = plt.gca()
    ax.set_facecolor('#FAFAFA')
    plt.gcf().set_facecolor('#FAFAFA')
    
    # Styling spine
    for spine in ax.spines.values():
        spine.set_edgecolor('#E0E0E0')
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    return plt

# === Aplikasi Utama ===
def main():
    # Header yang menawan
    st.markdown("""
    <div class="hero-header fade-in-up">
        <div class="logo-container">
            <img src="https://i.imgur.com/JXG8sK4.png" alt="Logo Xila Studio">
            <h1 class="hero-title">Xila Studio</h1>
            <img src="https://i.imgur.com/JXG8sK4.png" alt="Logo Xila Studio">
        </div>
        <p class="hero-subtitle">✨ Sistem Prediksi Stok Hoodie yang Elegan ✨</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar untuk navigasi
    st.sidebar.markdown("""
    <div style="padding: 20px; border-radius: 15px; background: rgba(255,255,255,0.1); backdrop-filter: blur(10px); margin-bottom: 30px;">
        <h2 style="color: white; font-family: 'Playfair Display', serif; text-align: center; margin-bottom: 20px;">
            🧭 Navigasi
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.selectbox("", ["📊 Dashboard", "🔮 Prediksi Stok", "📈 Analisis Penjualan", "ℹ️ Tentang"])
    
    # Cek data
    if not Path(DATA_PATH).exists():
        st.error("❌ File 'dataset.csv' tidak ditemukan di folder ini!")
        st.stop()
    
    # Muat data dengan animasi loading yang menarik
    with st.spinner("""
        <div class="loading-container">
            <div class="loading-spinner"></div>
        </div>
    """, unsafe_allow_html=True):
        df = load_and_clean_data(DATA_PATH)
    
    if df is None:
        st.stop()
    
    st.success(f"✅ Data berhasil dimuat: {len(df)} hari dari {df['Tanggal_Transaksi'].min().date()} hingga {df['Tanggal_Transaksi'].max().date()}")
    
    # Muat atau latih model
    if Path(MODEL_PATH).exists():
        model = joblib.load(MODEL_PATH)
        # Buat df_model untuk rata-rata fitur
        df_model = df.copy()
        df_model["hari_dalam_minggu"] = df_model["Tanggal_Transaksi"].dt.dayofweek
        df_model["bulan"] = df_model["Tanggal_Transaksi"].dt.month
        st.info("ℹ️ Model dimuat dari file yang sudah ada.")
    else:
        with st.spinner("""
            <div class="loading-container">
                <div class="loading-spinner"></div>
            </div>
        """, unsafe_allow_html=True):
            model, df_model = train_and_save_model(df)
        st.success("""
            <div class="success-animation"></div>
            <span style="margin-left: 10px; font-weight: bold;">✅ Model berhasil dilatih dan disimpan!</span>
        """, unsafe_allow_html=True)
    
    # Konten berdasarkan halaman yang dipilih
    if page == "📊 Dashboard":
        st.markdown("""
        <div class="elegant-card fade-in-up">
            <h2 style="color: var(--primary-color); font-family: 'Playfair Display', serif; margin-bottom: 30px;">
                📊 Dashboard Penjualan
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrik utama dengan tampilan yang menawan
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card fade-in-up">
                <div class="metric-value">{}</div>
                <div class="metric-label">🛍️ Total Penjualan</div>
            </div>
            """.format(f"{df['Produk_Terjual'].sum():,.0f}"), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card fade-in-up" style="animation-delay: 0.1s;">
                <div class="metric-value">{}</div>
                <div class="metric-label">📈 Rata-rata Harian</div>
            </div>
            """.format(f"{df['Produk_Terjual'].mean():,.1f}"), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card fade-in-up" style="animation-delay: 0.2s;">
                <div class="metric-value">{}</div>
                <div class="metric-label">👥 Total Pembeli</div>
            </div>
            """.format(f"{df['Pembeli'].sum():,.0f}"), unsafe_allow_html=True)
        
        with col4:
            konversi_values = pd.to_numeric(df['Persentase_Konversi'], errors='coerce')
            if konversi_values.notna().any():
                st.markdown("""
                <div class="metric-card fade-in-up" style="animation-delay: 0.3s;">
                    <div class="metric-value">{}</div>
                    <div class="metric-label">🔄 Konversi Rata-rata</div>
                </div>
                """.format(f"{konversi_values.mean():.2f}%"), unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="metric-card fade-in-up" style="animation-delay: 0.3s;">
                    <div class="metric-value">N/A</div>
                    <div class="metric-label">🔄 Konversi Rata-rata</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Grafik penjualan dengan container yang menawan
        st.markdown("""
        <div class="chart-container fade-in-up" style="animation-delay: 0.4s;">
            <h3 style="color: var(--primary-color); font-family: 'Playfair Display', serif; margin-bottom: 20px;">
                📈 Tren Penjualan
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            fig = plot_sales_data_daily(df)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Gagal menampilkan grafik: {str(e)}")
        
        # Tabel data terbaru
        st.markdown("""
        <div class="elegant-card fade-in-up" style="animation-delay: 0.5s;">
            <h3 style="color: var(--primary-color); font-family: 'Playfair Display', serif; margin-bottom: 20px;">
                📋 Data Penjualan Terbaru
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        df_display = df.tail(10).sort_values('Tanggal_Transaksi', ascending=False).copy()
        # Format tanggal tanpa jam
        df_display['Tanggal_Transaksi'] = df_display['Tanggal_Transaksi'].dt.strftime('%Y-%m-%d')
        st.dataframe(df_display)
    
    elif page == "🔮 Prediksi Stok":
        st.markdown("""
        <div class="elegant-card fade-in-up">
            <h2 style="color: var(--primary-color); font-family: 'Playfair Display', serif; margin-bottom: 30px;">
                🔮 Prediksi Kebutuhan Stok
            </h2>
            <p style="color: var(--text-color); font-size: 1.1rem; line-height: 1.6;">
                Sistem ini akan memprediksi kebutuhan stok hoodie untuk periode tertentu di masa depan dengan akurasi tinggi.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Pilihan tipe prediksi
        prediksi_type = st.radio("", ["📅 Harian", "📅 Bulanan"], horizontal=True)
        
        if prediksi_type == "📅 Harian":
            # Input pengguna untuk prediksi harian
            col1, col2 = st.columns(2)
            with col1:
                n_hari = st.slider("Jumlah hari ke depan", min_value=7, max_value=180, value=30, step=1)
            with col2:
                safety_pct = st.slider("Safety stock (%)", min_value=0, max_value=50, value=20, step=1)
            
            # Prediksi dengan button yang menarik
            if st.button("🚀 Prediksi Kebutuhan Stok Harian", key="prediksi_harian", help="Klik untuk memulai prediksi"):
                with st.spinner("""
                    <div class="loading-container">
                        <div class="loading-spinner"></div>
                    </div>
                """, unsafe_allow_html=True):
                    hasil, total_stok = predict_stock_daily(model, df_model, n_hari, safety_pct)
                
                # Hasil prediksi
                st.markdown(f"""
                <div class="elegant-card fade-in-up">
                    <h3 style="color: var(--primary-color); font-family: 'Playfair Display', serif; margin-bottom: 30px;">
                        📊 Hasil Prediksi untuk {n_hari} Hari ke Depan
                    </h3>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("""
                    <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
                        <div class="metric-value" style="color: white;">{}</div>
                        <div class="metric-label" style="color: rgba(255,255,255,0.9);">📦 Total Stok</div>
                    </div>
                    """.format(f"{total_stok:,.0f}"), unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white;">
                        <div class="metric-value" style="color: white;">{}</div>
                        <div class="metric-label" style="color: rgba(255,255,255,0.9);">📊 Rata-rata/Hari</div>
                    </div>
                    """.format(f"{hasil['Prediksi_Produk_Terjual'].mean():,.1f}"), unsafe_allow_html=True)
                
                with col3:
                    st.markdown("""
                    <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white;">
                        <div class="metric-value" style="color: white;">{}</div>
                        <div class="metric-label" style="color: rgba(255,255,255,0.9);">🛡️ Safety Stock</div>
                    </div>
                    """.format(f"{total_stok - hasil['Prediksi_Produk_Terjual'].mean() * n_hari:,.1f}"), unsafe_allow_html=True)
                
                # Grafik prediksi
                st.markdown("""
                <div class="chart-container fade-in-up">
                    <h3 style="color: var(--primary-color); font-family: 'Playfair Display', serif; margin-bottom: 20px;">
                        📈 Grafik Prediksi Penjualan
                    </h3>
                </div>
                """, unsafe_allow_html=True)
                
                try:
                    fig = plot_sales_data_daily(df, hasil)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Gagal menampilkan grafik prediksi: {str(e)}")
                
                # Tabel prediksi
                st.markdown("""
                <div class="elegant-card fade-in-up">
                    <h3 style="color: var(--primary-color); font-family: 'Playfair Display', serif; margin-bottom: 20px;">
                        📋 Detail Prediksi Harian
                    </h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Format tanggal tanpa jam
                hasil_display = hasil.copy()
                hasil_display['Tanggal'] = hasil_display['Tanggal'].dt.strftime('%Y-%m-%d')
                st.dataframe(hasil_display.set_index('Tanggal'))
                
                # Rekomendasi pemesanan
                st.markdown("""
                <div class="elegant-card fade-in-up" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); color: white;">
                    <h3 style="color: white; font-family: 'Playfair Display', serif; margin-bottom: 20px;">
                        💡 Rekomendasi Pemesanan
                    </h3>
                    <p style="font-size: 1.1rem; line-height: 1.6;">
                        Disarankan untuk memesan minimal <strong>{total_stok:,.0f} unit</strong> hoodie untuk memenuhi permintaan dalam {n_hari} hari ke depan dengan safety stock {safety_pct}%.
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        else:  # Bulanan
            # Input pengguna untuk prediksi bulanan
            col1, col2 = st.columns(2)
            with col1:
                n_bulan = st.slider("Jumlah bulan ke depan", min_value=1, max_value=12, value=3, step=1)
            with col2:
                safety_pct = st.slider("Safety stock (%)", min_value=0, max_value=50, value=20, step=1)
            
            # Prediksi
            if st.button("🚀 Prediksi Kebutuhan Stok Bulanan", key="prediksi_bulanan"):
                with st.spinner("""
                    <div class="loading-container">
                        <div class="loading-spinner"></div>
                    </div>
                """, unsafe_allow_html=True):
                    hasil, total_stok = predict_stock_monthly(model, df_model, n_bulan, safety_pct)
                
                # Hasil prediksi
                st.markdown(f"""
                <div class="elegant-card fade-in-up">
                    <h3 style="color: var(--primary-color); font-family: 'Playfair Display', serif; margin-bottom: 30px;">
                        📊 Hasil Prediksi untuk {n_bulan} Bulan ke Depan
                    </h3>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("""
                    <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
                        <div class="metric-value" style="color: white;">{}</div>
                        <div class="metric-label" style="color: rgba(255,255,255,0.9);">📦 Total Stok</div>
                    </div>
                    """.format(f"{total_stok:,.0f}"), unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white;">
                        <div class="metric-value" style="color: white;">{}</div>
                        <div class="metric-label" style="color: rgba(255,255,255,0.9);">📊 Rata-rata/Bulan</div>
                    </div>
                    """.format(f"{hasil['Prediksi_Produk_Terjual'].mean():,.1f}"), unsafe_allow_html=True)
                
                with col3:
                    st.markdown("""
                    <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white;">
                        <div class="metric-value" style="color: white;">{}</div>
                        <div class="metric-label" style="color: rgba(255,255,255,0.9);">🛡️ Safety Stock</div>
                    </div>
                    """.format(f"{total_stok - hasil['Prediksi_Produk_Terjual'].sum():,.1f}"), unsafe_allow_html=True)
                
                # Grafik prediksi
                st.markdown("""
                <div class="chart-container fade-in-up">
                    <h3 style="color: var(--primary-color); font-family: 'Playfair Display', serif; margin-bottom: 20px;">
                        📈 Grafik Prediksi Penjualan
                    </h3>
                </div>
                """, unsafe_allow_html=True)
                
                try:
                    fig = plot_sales_data_monthly(df, hasil)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Gagal menampilkan grafik prediksi: {str(e)}")
                
                # Tabel prediksi
                st.markdown("""
                <div class="elegant-card fade-in-up">
                    <h3 style="color: var(--primary-color); font-family: 'Playfair Display', serif; margin-bottom: 20px;">
                        📋 Detail Prediksi Bulanan
                    </h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.dataframe(hasil.set_index('Bulan'))
                
                # Rekomendasi pemesanan
                st.markdown("""
                <div class="elegant-card fade-in-up" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); color: white;">
                    <h3 style="color: white; font-family: 'Playfair Display', serif; margin-bottom: 20px;">
                        💡 Rekomendasi Pemesanan
                    </h3>
                    <p style="font-size: 1.1rem; line-height: 1.6;">
                        Disarankan untuk memesan minimal <strong>{total_stok:,.0f} unit</strong> hoodie untuk memenuhi permintaan dalam {n_bulan} bulan ke depan dengan safety stock {safety_pct}%.
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    elif page == "📈 Analisis Penjualan":
        st.markdown("""
        <div class="elegant-card fade-in-up">
            <h2 style="color: var(--primary-color); font-family: 'Playfair Display', serif; margin-bottom: 30px;">
                📈 Analisis Penjualan
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Pilihan analisis - Korelasi Fitur sudah dihapus
        analisis_type = st.selectbox("Pilih Jenis Analisis", ["📅 Penjualan per Bulan", "📆 Penjualan per Hari"])
        
        if analisis_type == "📅 Penjualan per Bulan":
            df_bulanan = df.copy()
            df_bulanan['Bulan'] = df_bulanan['Tanggal_Transaksi'].dt.to_period('M')
            df_bulanan = df_bulanan.groupby('Bulan').agg({
                'Produk_Terjual': 'sum',
                'Pembeli': 'sum',
                'Pesanan': 'sum'
            }).reset_index()
            df_bulanan['Bulan'] = df_bulanan['Bulan'].astype(str)
            
            st.markdown("""
            <div class="elegant-card fade-in-up">
                <h3 style="color: var(--primary-color); font-family: 'Playfair Display', serif; margin-bottom: 20px;">
                    📊 Penjualan per Bulan
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.dataframe(df_bulanan)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(df_bulanan['Bulan'], df_bulanan['Produk_Terjual'], color='#6C5CE7')
            plt.title('Total Penjualan per Bulan', fontweight='bold', pad=20)
            plt.xlabel('Bulan', fontsize=12, labelpad=10)
            plt.ylabel('Jumlah Terjual', fontsize=12, labelpad=10)
            plt.xticks(rotation=45)
            ax.set_facecolor('#FAFAFA')
            plt.gcf().set_facecolor('#FAFAFA')
            plt.tight_layout()
            st.pyplot(fig)
        
        elif analisis_type == "📆 Penjualan per Hari":
            df['Hari'] = df['Tanggal_Transaksi'].dt.day_name()
            df_harian = df.groupby('Hari')['Produk_Terjual'].mean().reset_index()
            
            # Urutkan hari
            hari_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            df_harian['Hari'] = pd.Categorical(df_harian['Hari'], categories=hari_order, ordered=True)
            df_harian = df_harian.sort_values('Hari')
            
            st.markdown("""
            <div class="elegant-card fade-in-up">
                <h3 style="color: var(--primary-color); font-family: 'Playfair Display', serif; margin-bottom: 20px;">
                    📊 Rata-rata Penjualan per Hari
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.dataframe(df_harian)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(df_harian['Hari'], df_harian['Produk_Terjual'], color='#FECA57')
            plt.title('Rata-rata Penjualan per Hari', fontweight='bold', pad=20)
            plt.xlabel('Hari', fontsize=12, labelpad=10)
            plt.ylabel('Rata-rata Jumlah Terjual', fontsize=12, labelpad=10)
            plt.xticks(rotation=45)
            ax.set_facecolor('#FAFAFA')
            plt.gcf().set_facecolor('#FAFAFA')
            plt.tight_layout()
            st.pyplot(fig)
    
    elif page == "ℹ️ Tentang":
        st.markdown("""
        <div class="elegant-card fade-in-up">
            <h2 style="color: var(--primary-color); font-family: 'Playfair Display', serif; margin-bottom: 30px;">
                ℹ️ Tentang Xila Studio
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            <div class="fade-in-up">
                <h3 style="color: var(--primary-color); font-family: 'Playfair Display', serif; margin-bottom: 20px;">
                    🏪 Tentang Xila Studio
                </h3>
                <p style="font-size: 1.1rem; line-height: 1.6; color: var(--text-color);">
                    <strong style="font-size: 1.3rem; color: var(--primary-color);">Xila Studio</strong> adalah toko online premium yang berfokus pada penjualan hoodie berkualitas tinggi dengan desain unik dan trendy. Setiap produk dibuat dengan cinta dan perhatian terhadap detail, menciptakan pengalaman berbelanja yang tak terlupakan.
                </p>
                
                <h4 style="color: var(--secondary-color); font-family: 'Playfair Display', serif; margin: 25px 0 15px 0;">
                    🌟 Visi Kami
                </h4>
                <p style="font-size: 1.1rem; line-height: 1.6; color: var(--text-color);">
                    Menjadi merek pilihan utama untuk hoodie yang nyaman, stylish, dan terjangkau di kalangan anak muda yang dinamis dan berjiwa tinggi.
                </p>
                
                <h4 style="color: var(--secondary-color); font-family: 'Playfair Display', serif; margin: 25px 0 15px 0;">
                    🎯 Misi Kami
                </h4>
                <ul style="font-size: 1.1rem; line-height: 1.8; color: var(--text-color); padding-left: 20px;">
                    <li>🧵 Menyediakan hoodie berkualitas premium dengan desain terkini</li>
                    <li>🛍️ Memberikan pengalaman berbelanja yang menyenangkan dan personal</li>
                    <li>🤝 Membangun komunitas pecinta hoodie yang solid</li>
                    <li>♻️ Mendukung industri fashion lokal yang berkelanjutan</li>
                </ul>
                
                <h4 style="color: var(--secondary-color); font-family: 'Playfair Display', serif; margin: 25px 0 15px 0;">
                    🤖 Tentang Sistem Ini
                </h4>
                <p style="font-size: 1.1rem; line-height: 1.6; color: var(--text-color);">
                    Sistem prediksi stok canggih ini dikembangkan dengan teknologi AI terdepan untuk membantu Xila Studio dalam:
                </p>
                <ul style="font-size: 1.1rem; line-height: 1.8; color: var(--text-color); padding-left: 20px;">
                    <li>📊 Memperkirakan kebutuhan stok hoodie dengan akurasi tinggi</li>
                    <li>📦 Mengoptimalkan persediaan barang secara efisien</li>
                    <li>⚖️ Mengurangi risiko kehabisan stok atau kelebihan stok</li>
                    <li>🚀 Meningkatkan efisiensi rantai pasokan</li>
                    <li>💰 Mengoptimalkan profitabilitas bisnis</li>
                </ul>
                
                <h4 style="color: var(--secondary-color); font-family: 'Playfair Display', serif; margin: 25px 0 15px 0;">
                    ✨ Keunggulan Produk
                </h4>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-top: 20px;">
                    <div style="background: rgba(108, 92, 231, 0.1); padding: 20px; border-radius: 15px;">
                        <h5 style="color: var(--primary-color); margin-bottom: 10px;">🧵 Kualitas Premium</h5>
                        <p>Bahan terbaik dengan standar kualitas internasional</p>
                    </div>
                    <div style="background: rgba(162, 155, 254, 0.1); padding: 20px; border-radius: 15px;">
                        <h5 style="color: var(--primary-color); margin-bottom: 10px;">🎨 Desain Eksklusif</h5>
                        <p>Desain unik yang tidak akan Anda temukan di tempat lain</p>
                    </div>
                    <div style="background: rgba(254, 202, 87, 0.1); padding: 20px; border-radius: 15px;">
                        <h5 style="color: var(--primary-color); margin-bottom: 10px;">💰 Harga Kompetitif</h5>
                        <p>Kualitas premium dengan harga yang terjangkau</p>
                    </div>
                    <div style="background: rgba(67, 172, 254, 0.1); padding: 20px; border-radius: 15px;">
                        <h5 style="color: var(--primary-color); margin-bottom: 10px;">🚚 Pengiriman Cepat</h5>
                        <p>Pengiriman tepat waktu ke seluruh Indonesia</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="text-align: center; margin-bottom: 30px;">
                <img src="https://i.imgur.com/5Nv5Q6e.jpg" width="300" style="border-radius: 20px; box-shadow: 0 15px 35px rgba(0,0,0,0.2);">
            </div>
            
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 20px; text-align: center;">
                <h3 style="font-family: 'Playfair Display', serif; margin-bottom: 20px;">
                    🌟 Kenapa Memilih Kami?
                </h3>
                <ul style="text-align: left; font-size: 1rem; line-height: 1.8; list-style: none; padding: 0;">
                    <li style="margin-bottom: 15px;">✨ Produk berkualitas tinggi</li>
                    <li style="margin-bottom: 15px;">🎨 Desain eksklusif dan trendy</li>
                    <li style="margin-bottom: 15px;">🛍️ Harga yang kompetitif</li>
                    <li style="margin-bottom: 15px;">🚚 Pengiriman cepat dan aman</li>
                    <li style="margin-bottom: 15px;">💬 Pelayanan pelanggan premium</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
