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

# === Konfigurasi Tema Profesional dengan Animasi ===
st.set_page_config(
    page_title="Xila Studio - Sistem Prediksi Stok Hoodie", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk tema profesional dengan animasi
st.markdown("""
<style>
    :root {
        --primary-color: #2E86AB;
        --secondary-color: #A23B72;
        --accent-color: #F18F01;
        --dark-color: #C73E1D;
        --light-color: #FFFFFF;
        --neutral-color: #F5F5F5;
        --text-color: #333333;
        --sidebar-color: #1a5276;
        --sidebar-text-color: #FFFFFF;
    }
    
    .stApp {
        background-color: var(--neutral-color);
    }
    
    /* Animasi Fade In */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Animasi Slide In dari kiri */
    @keyframes slideInLeft {
        from { transform: translateX(-100%); }
        to { transform: translateX(0); }
    }
    
    /* Animasi Pulse */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    /* Animasi Bounce */
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    /* Animasi Glow */
    @keyframes glow {
        0% { box-shadow: 0 0 5px rgba(46, 134, 171, 0.5); }
        50% { box-shadow: 0 0 20px rgba(46, 134, 171, 0.8); }
        100% { box-shadow: 0 0 5px rgba(46, 134, 171, 0.5); }
    }
    
    /* Terapkan animasi ke elemen */
    .fade-in {
        animation: fadeIn 1s ease-out;
    }
    
    .slide-in {
        animation: slideInLeft 0.8s ease-out;
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    .bounce {
        animation: bounce 2s infinite;
    }
    
    .glow {
        animation: glow 2s infinite;
    }
    
    .st-bq {
        border-left: 4px solid var(--primary-color);
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 4px;
        animation: fadeIn 1.2s ease-out;
    }
    
    .stMetric {
        background-color: var(--light-color);
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid var(--primary-color);
        transition: all 0.3s ease;
        animation: fadeIn 1s ease-out;
    }
    
    .stMetric:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
        border-left: 6px solid var(--primary-color);
    }
    
    .stAlert {
        background-color: rgba(255, 255, 255, 0.9);
        border-left: 4px solid var(--primary-color);
        border-radius: 4px;
        animation: fadeIn 1s ease-out;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: var(--dark-color);
        font-family: 'Segoe UI', 'Roboto', sans-serif;
        font-weight: 600;
        animation: fadeIn 1.2s ease-out;
    }
    
    .stDataFrame {
        background-color: var(--light-color);
        border-radius: 4px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border: 1px solid #E0E0E0;
        animation: fadeIn 1.5s ease-out;
    }
    
    .stButton>button {
        background-color: var(--primary-color);
        color: var(--light-color);
        border: none;
        border-radius: 4px;
        padding: 10px 20px;
        font-weight: 500;
        font-family: 'Segoe UI', 'Roboto', sans-serif;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        animation: fadeIn 1s ease-out;
    }
    
    .stButton>button:before {
        content: "";
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.5s;
    }
    
    .stButton>button:hover:before {
        left: 100%;
    }
    
    .stButton>button:hover {
        background-color: #245A8F;
        color: var(--light-color);
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }
    
    .stSidebar {
        background-color: var(--sidebar-color);
        background-image: linear-gradient(180deg, var(--sidebar-color), #2c3e50);
        color: var(--sidebar-text-color);
        animation: slideInLeft 0.8s ease-out;
    }
    
    .stSidebar .stSelectbox>div>div>div {
        color: var(--sidebar-text-color);
        font-family: 'Segoe UI', 'Roboto', sans-serif;
    }
    
    .stSidebar .stSelectbox>div>div>div>div {
        background-color: rgba(255, 255, 255, 0.9);
        color: #333333;
        border-radius: 4px;
        transition: all 0.3s ease;
    }
    
    .stSidebar .stSelectbox>div>div>div>div:hover {
        background-color: rgba(255, 255, 255, 1);
        transform: scale(1.02);
    }
    
    .stSidebar .stSlider>div>div>div>div {
        background-color: var(--accent-color);
        border-radius: 4px;
        transition: all 0.3s ease;
    }
    
    .stSidebar .stSlider>div>div>div>div:hover {
        transform: scale(1.05);
    }
    
    .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4, .stSidebar h5, .stSidebar h6 {
        color: var(--sidebar-text-color);
    }
    
    .stSidebar label {
        color: var(--sidebar-text-color);
    }
    
    div[data-testid="stMetricValue"] {
        color: var(--primary-color);
        font-size: 1.6rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    div[data-testid="stMetricLabel"] {
        color: var(--text-color);
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .element-container img {
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        border: 2px solid var(--light-color);
        transition: all 0.3s ease;
        animation: fadeIn 1s ease-out;
    }
    
    .element-container img:hover {
        transform: scale(1.03);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }
    
    .streamlit-container {
        max-width: 1200px;
        padding-top: 1rem;
    }
    
    /* Professional styling for headers */
    .main-header {
        border-bottom: 2px solid var(--primary-color);
        padding-bottom: 10px;
        margin-bottom: 20px;
        position: relative;
        animation: fadeIn 1s ease-out;
    }
    
    .main-header:after {
        content: "";
        position: absolute;
        bottom: -2px;
        left: 0;
        width: 100px;
        height: 2px;
        background-color: var(--accent-color);
        animation: expandWidth 2s ease-out;
    }
    
    @keyframes expandWidth {
        from { width: 0; }
        to { width: 100px; }
    }
    
    /* Card styling */
    .card {
        background-color: var(--light-color);
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        border-left: 4px solid var(--primary-color);
        animation: fadeIn 1.2s ease-out;
        transition: all 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Loading animation */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(255,255,255,.3);
        border-radius: 50%;
        border-top-color: var(--primary-color);
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Success animation */
    .success-checkmark {
        display: inline-block;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        background-color: var(--accent-color);
        position: relative;
        animation: checkmark 0.6s ease-in-out;
    }
    
    @keyframes checkmark {
        0% { transform: scale(0); }
        50% { transform: scale(1.2); }
        100% { transform: scale(1); }
    }
    
    .success-checkmark:after {
        content: "✓";
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        color: white;
        font-weight: bold;
    }
    
    /* White button styling */
    .white-button {
        background-color: #FFFFFF;
        color: #2E86AB;
        border: 1px solid #2E86AB;
        border-radius: 4px;
        padding: 8px 16px;
        font-weight: 500;
        font-family: 'Segoe UI', 'Roboto', sans-serif;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        cursor: pointer;
        display: inline-block;
        margin: 5px 0;
    }
    
    .white-button:hover {
        background-color: #f8f9fa;
        color: #245A8F;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
</style>
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
    plt.figure(figsize=(12, 6))
    
    # Plot data historis dengan warna profesional
    plt.plot(df['Tanggal_Transaksi'], df['Produk_Terjual'], 
             label='Data Historis', color='#2E86AB', linewidth=2.5)
    
    # Plot prediksi dengan warna profesional lainnya
    if hasil_prediksi is not None:
        plt.plot(hasil_prediksi['Tanggal'], hasil_prediksi['Prediksi_Produk_Terjual'], 
                 label='Prediksi', color='#A23B72', linewidth=2.5, linestyle='--')
    
    plt.title('Grafik Penjualan Harian', fontsize=16, fontweight='bold')
    plt.xlabel('Tanggal', fontsize=12)
    plt.ylabel('Jumlah Terjual', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(facecolor='white', framealpha=0.9, edgecolor='#2E86AB')
    plt.tight_layout()
    
    # Mengatur warna background grafik
    ax = plt.gca()
    ax.set_facecolor('#FFFFFF')
    plt.gcf().set_facecolor('#FFFFFF')
    
    # Mengatur warna spine
    for spine in ax.spines.values():
        spine.set_edgecolor('#E0E0E0')
        spine.set_linewidth(1)
    
    return plt

# === Fungsi: Visualisasi Data Bulanan ===
def plot_sales_data_monthly(df, hasil_prediksi=None):
    plt.figure(figsize=(12, 6))
    
    # Prepare monthly historical data
    df_bulanan = df.copy()
    df_bulanan['Bulan'] = df_bulanan['Tanggal_Transaksi'].dt.to_period('M')
    df_bulanan = df_bulanan.groupby('Bulan')['Produk_Terjual'].sum().reset_index()
    df_bulanan['Bulan'] = df_bulanan['Bulan'].astype(str)
    
    # Plot data historis
    plt.bar(df_bulanan['Bulan'], df_bulanan['Produk_Terjual'], 
             label='Data Historis', color='#2E86AB', alpha=0.7)
    
    # Plot prediksi jika ada
    if hasil_prediksi is not None:
        plt.bar(hasil_prediksi['Bulan'], hasil_prediksi['Prediksi_Produk_Terjual'], 
                 label='Prediksi', color='#A23B72', alpha=0.7)
    
    plt.title('Grafik Penjualan Bulanan', fontsize=16, fontweight='bold')
    plt.xlabel('Bulan', fontsize=12)
    plt.ylabel('Jumlah Terjual', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend(facecolor='white', framealpha=0.9, edgecolor='#2E86AB')
    plt.tight_layout()
    
    # Mengatur warna background grafik
    ax = plt.gca()
    ax.set_facecolor('#FFFFFF')
    plt.gcf().set_facecolor('#FFFFFF')
    
    # Mengatur warna spine
    for spine in ax.spines.values():
        spine.set_edgecolor('#E0E0E0')
        spine.set_linewidth(1)
    
    return plt

# === Aplikasi Utama ===
def main():
    # Header dengan logo dan nama toko
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        st.image("logo xila.jpg", width=120)  # Logo Xila Studio
    with col2:
        st.markdown("<h1 class='main-header fade-in'>Xila Studio</h1>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center;' class='fade-in'>Sistem Prediksi Stok Hoodie</h2>", unsafe_allow_html=True)
    with col3:
        st.image("logo xila.jpg", width=120)  # Logo hoodie
    
    st.markdown("---")
    
    # Sidebar untuk navigasi
    st.sidebar.title("Navigasi")
    page = st.sidebar.selectbox("Pilih Halaman", ["Dashboard", "Prediksi Stok", "Analisis Penjualan", "Tentang"])
    
    # Cek data
    if not Path(DATA_PATH).exists():
        st.error("File 'dataset.csv' tidak ditemukan di folder ini!")
        st.stop()
    
    # Muat data
    df = load_and_clean_data(DATA_PATH)
    if df is None:
        st.stop()
    
    st.markdown(f"<div class='fade-in'>✅ Data berhasil dimuat: {len(df)} hari dari {df['Tanggal_Transaksi'].min().date()} hingga {df['Tanggal_Transaksi'].max().date()}</div>", unsafe_allow_html=True)
    
    # Muat atau latih model
    if Path(MODEL_PATH).exists():
        model = joblib.load(MODEL_PATH)
        # Buat df_model untuk rata-rata fitur
        df_model = df.copy()
        df_model["hari_dalam_minggu"] = df_model["Tanggal_Transaksi"].dt.dayofweek
        df_model["bulan"] = df_model["Tanggal_Transaksi"].dt.month
        st.markdown("<div class='fade-in'>ℹ️ Model dimuat dari file yang sudah ada.</div>", unsafe_allow_html=True)
    else:
        with st.spinner("Melatih model..."):
            model, df_model = train_and_save_model(df)
        st.markdown("<div class='success-checkmark'></div> <span class='fade-in'>✅ Model berhasil dilatih dan disimpan!</span>", unsafe_allow_html=True)
    
    # Konten berdasarkan halaman yang dipilih
    if page == "Dashboard":
        st.markdown("<h2 class='fade-in'>Dashboard Penjualan</h2>", unsafe_allow_html=True)
        
        # Metrik utama
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Penjualan", f"{df['Produk_Terjual'].sum():,.0f} unit")
        with col2:
            st.metric("Rata-rata Harian", f"{df['Produk_Terjual'].mean():,.1f} unit")
        with col3:
            st.metric("Total Pembeli", f"{df['Pembeli'].sum():,.0f} orang")
        with col4:
            # Konversi Persentase_Konversi ke float untuk perhitungan rata-rata
            konversi_values = pd.to_numeric(df['Persentase_Konversi'], errors='coerce')
            if konversi_values.notna().any():
                st.metric("Konversi Rata-rata", f"{konversi_values.mean():.2f}%")
            else:
                st.metric("Konversi Rata-rata", "N/A")
        
        # Grafik penjualan
        st.markdown("<h3 class='fade-in'>Tren Penjualan</h3>", unsafe_allow_html=True)
        try:
            fig = plot_sales_data_daily(df)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Gagal menampilkan grafik: {str(e)}")
        
        # Tabel data terbaru
        st.markdown("<h3 class='fade-in'>Data Penjualan Terbaru</h3>", unsafe_allow_html=True)
        df_display = df.tail(10).sort_values('Tanggal_Transaksi', ascending=False).copy()
        # Format tanggal tanpa jam
        df_display['Tanggal_Transaksi'] = df_display['Tanggal_Transaksi'].dt.strftime('%Y-%m-%d')
        st.dataframe(df_display)
    
    elif page == "Prediksi Stok":
        st.markdown("<h2 class='fade-in'>Prediksi Kebutuhan Stok</h2>", unsafe_allow_html=True)
        st.markdown("<div class='fade-in'>Sistem ini akan memprediksi kebutuhan stok hoodie untuk periode tertentu di masa depan.</div>", unsafe_allow_html=True)
        
        # Pilihan tipe prediksi
        prediksi_type = st.radio("Pilih Tipe Prediksi", ["Harian", "Bulanan"], horizontal=True)
        
        if prediksi_type == "Harian":
            # Input pengguna untuk prediksi harian
            col1, col2 = st.columns(2)
            with col1:
                n_hari = st.slider("Jumlah hari ke depan", min_value=7, max_value=180, value=30, step=1)
            with col2:
                safety_pct = st.slider("Safety stock (%)", min_value=0, max_value=50, value=20, step=1)
            
            # Prediksi
            if st.button("Prediksi Kebutuhan Stok Harian", use_container_width=True):
                with st.spinner("Memproses prediksi..."):
                    hasil, total_stok = predict_stock_daily(model, df_model, n_hari, safety_pct)
                
                # Hasil prediksi
                st.markdown(f"<h3 class='fade-in'>Hasil Prediksi untuk {n_hari} Hari ke Depan</h3>", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Stok yang Disarankan", f"{total_stok:,.0f} unit")
                with col2:
                    st.metric("Rata-rata Penjualan/Hari", f"{hasil['Prediksi_Produk_Terjual'].mean():,.1f} unit")
                with col3:
                    st.metric("Safety Stock", f"{total_stok - hasil['Prediksi_Produk_Terjual'].mean() * n_hari:,.1f} unit")
                
                # Grafik prediksi
                st.markdown("<h3 class='fade-in'>Grafik Prediksi Penjualan</h3>", unsafe_allow_html=True)
                try:
                    fig = plot_sales_data_daily(df, hasil)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Gagal menampilkan grafik prediksi: {str(e)}")
                
                # Tabel prediksi
                st.markdown("<h3 class='fade-in'>Detail Prediksi Harian</h3>", unsafe_allow_html=True)
                # Format tanggal tanpa jam
                hasil_display = hasil.copy()
                hasil_display['Tanggal'] = hasil_display['Tanggal'].dt.strftime('%Y-%m-%d')
                st.dataframe(hasil_display.set_index('Tanggal'))
                
                # Rekomendasi pemesanan
                st.markdown("<h3 class='fade-in'>Rekomendasi Pemesanan</h3>", unsafe_allow_html=True)
                st.info(f"Disarankan untuk memesan minimal **{total_stok:,.0f} unit** hoodie untuk memenuhi permintaan dalam {n_hari} hari ke depan dengan safety stock {safety_pct}%.")
        
        else:  # Bulanan
            # Input pengguna untuk prediksi bulanan
            col1, col2 = st.columns(2)
            with col1:
                n_bulan = st.slider("Jumlah bulan ke depan", min_value=1, max_value=12, value=3, step=1)
            with col2:
                safety_pct = st.slider("Safety stock (%)", min_value=0, max_value=50, value=20, step=1)
            
            # Prediksi
            if st.button("Prediksi Kebutuhan Stok Bulanan", use_container_width=True):
                with st.spinner("Memproses prediksi..."):
                    hasil, total_stok = predict_stock_monthly(model, df_model, n_bulan, safety_pct)
                
                # Hasil prediksi
                st.markdown(f"<h3 class='fade-in'>Hasil Prediksi untuk {n_bulan} Bulan ke Depan</h3>", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Stok yang Disarankan", f"{total_stok:,.0f} unit")
                with col2:
                    st.metric("Rata-rata Penjualan/Bulan", f"{hasil['Prediksi_Produk_Terjual'].mean():,.1f} unit")
                with col3:
                    st.metric("Safety Stock", f"{total_stok - hasil['Prediksi_Produk_Terjual'].sum():,.1f} unit")
                
                # Grafik prediksi
                st.markdown("<h3 class='fade-in'>Grafik Prediksi Penjualan</h3>", unsafe_allow_html=True)
                try:
                    fig = plot_sales_data_monthly(df, hasil)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Gagal menampilkan grafik prediksi: {str(e)}")
                
                # Tabel prediksi
                st.markdown("<h3 class='fade-in'>Detail Prediksi Bulanan</h3>", unsafe_allow_html=True)
                st.dataframe(hasil.set_index('Bulan'))
                
                # Rekomendasi pemesanan
                st.markdown("<h3 class='fade-in'>Rekomendasi Pemesanan</h3>", unsafe_allow_html=True)
                st.info(f"Disarankan untuk memesan minimal **{total_stok:,.0f} unit** hoodie untuk memenuhi permintaan dalam {n_bulan} bulan ke depan dengan safety stock {safety_pct}%.")
    
    elif page == "Analisis Penjualan":
        st.markdown("<h2 class='fade-in'>Analisis Penjualan</h2>", unsafe_allow_html=True)
        
        # Pilihan analisis - Korelasi Fitur sudah dihapus
        analisis_type = st.selectbox("Pilih Jenis Analisis", ["Penjualan per Bulan", "Penjualan per Hari"])
        
        if analisis_type == "Penjualan per Bulan":
            df_bulanan = df.copy()
            df_bulanan['Bulan'] = df_bulanan['Tanggal_Transaksi'].dt.to_period('M')
            df_bulanan = df_bulanan.groupby('Bulan').agg({
                'Produk_Terjual': 'sum',
                'Pembeli': 'sum',
                'Pesanan': 'sum'
            }).reset_index()
            df_bulanan['Bulan'] = df_bulanan['Bulan'].astype(str)
            
            st.markdown("<h3 class='fade-in'>Penjualan per Bulan</h3>", unsafe_allow_html=True)
            st.dataframe(df_bulanan)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(df_bulanan['Bulan'], df_bulanan['Produk_Terjual'], color='#2E86AB')
            plt.title('Total Penjualan per Bulan', fontweight='bold')
            plt.xlabel('Bulan')
            plt.ylabel('Jumlah Terjual')
            plt.xticks(rotation=45)
            ax.set_facecolor('#FFFFFF')
            plt.gcf().set_facecolor('#FFFFFF')
            plt.tight_layout()
            st.pyplot(fig)
        
        elif analisis_type == "Penjualan per Hari":
            df['Hari'] = df['Tanggal_Transaksi'].dt.day_name()
            df_harian = df.groupby('Hari')['Produk_Terjual'].mean().reset_index()
            
            # Urutkan hari
            hari_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            df_harian['Hari'] = pd.Categorical(df_harian['Hari'], categories=hari_order, ordered=True)
            df_harian = df_harian.sort_values('Hari')
            
            st.markdown("<h3 class='fade-in'>Rata-rata Penjualan per Hari</h3>", unsafe_allow_html=True)
            st.dataframe(df_harian)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(df_harian['Hari'], df_harian['Produk_Terjual'], color='#A23B72')
            plt.title('Rata-rata Penjualan per Hari', fontweight='bold')
            plt.xlabel('Hari')
            plt.ylabel('Rata-rata Jumlah Terjual')
            plt.xticks(rotation=45)
            ax.set_facecolor('#FFFFFF')
            plt.gcf().set_facecolor('#FFFFFF')
            plt.tight_layout()
            st.pyplot(fig)
    
    elif page == "Tentang":
        st.markdown("<h2 class='fade-in'>Tentang Xila Studio</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            <div class="fade-in">
            **Xila Studio** adalah toko online yang berfokus pada penjualan hoodie berkualitas tinggi dengan desain unik dan trendy.
            
            ### Visi
            Menjadi merek pilihan utama untuk hoodie yang nyaman, stylish, dan terjangkau di kalangan anak muda.
            
            ### Misi
            - Menyediakan hoodie berkualitas dengan desain terkini
            - Memberikan pengalaman berbelanja yang menyenangkan
            - Membangun komunitas pecinta hoodie
            
            ### Tentang Sistem Ini
            Sistem prediksi stok ini dikembangkan untuk membantu Xila Studio dalam:
            - Memperkirakan kebutuhan stok hoodie untuk periode tertentu
            - Mengoptimalkan persediaan barang
            - Mengurangi risiko kehabisan stok atau kelebihan stok
            - Meningkatkan efisiensi rantai pasokan
            
            ### Keunggulan Produk
            - Kualitas bahan terjamin
            - Desain eksklusif dan trendy
            - Harga kompetitif
            - Pengiriman tepat waktu
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.image("logo xila.jpg", width=300)  # Gambar toko atau produk

if __name__ == "__main__":
    main()