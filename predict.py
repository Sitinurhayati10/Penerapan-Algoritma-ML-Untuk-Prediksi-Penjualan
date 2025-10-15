# =========================================================
# main.py - Sistem Prediksi Stok Hoodie | Xila Studio
# =========================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns

# === Konfigurasi Tema Profesional dengan Animasi ===
st.set_page_config(
    page_title="Xila Studio - Sistem Prediksi Stok Hoodie",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tambahan styling CSS (tema, animasi, tampilan elegan)
st.markdown("""<style>
/* Gaya CSS sudah panjang, tetap dipakai dari versi awalmu */
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
/* Animasi dan elemen UI */
.stApp {background-color: var(--neutral-color);}
h1,h2,h3,h4,h5,h6 {color: var(--dark-color);font-weight:600;}
.stButton>button{background-color:var(--primary-color);color:white;}
.stSidebar{background-color:var(--sidebar-color);}
div[data-testid="stMetricValue"] {color: var(--primary-color);}
</style>""", unsafe_allow_html=True)

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

# === Fungsi: Plot Grafik Penjualan Harian ===
def plot_sales_data_daily(df, hasil_prediksi=None):
    plt.figure(figsize=(12, 6))
    plt.plot(df['Tanggal_Transaksi'], df['Produk_Terjual'], label='Data Historis', color='#2E86AB', linewidth=2.5)
    if hasil_prediksi is not None:
        plt.plot(hasil_prediksi['Tanggal'], hasil_prediksi['Prediksi_Produk_Terjual'], label='Prediksi', color='#A23B72', linestyle='--')
    plt.title('Grafik Penjualan Harian', fontsize=16, fontweight='bold')
    plt.xlabel('Tanggal')
    plt.ylabel('Jumlah Terjual')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    return plt

# === Aplikasi Utama ===
def main():
    # Header
    col1, col2, col3 = st.columns([1,3,1])
    with col1: st.image("logo xila.jpg", width=120)
    with col2:
        st.markdown("<h1 style='text-align:center'>Xila Studio</h1>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align:center'>Sistem Prediksi Stok Hoodie</h2>", unsafe_allow_html=True)
    with col3: st.image("logo xila.jpg", width=120)
    st.markdown("---")

    # Sidebar
    st.sidebar.title("Navigasi")
    page = st.sidebar.selectbox("Pilih Halaman", ["Dashboard", "Prediksi Stok", "Analisis Penjualan", "Tentang"])

    # Load Data
    if not Path(DATA_PATH).exists():
        st.error("File 'dataset.csv' tidak ditemukan!")
        st.stop()

    df = load_and_clean_data(DATA_PATH)
    if df is None:
        st.stop()

    if Path(MODEL_PATH).exists():
        model = joblib.load(MODEL_PATH)
        df_model = df.copy()
        df_model["hari_dalam_minggu"] = df_model["Tanggal_Transaksi"].dt.dayofweek
        df_model["bulan"] = df_model["Tanggal_Transaksi"].dt.month
    else:
        with st.spinner("Melatih model..."):
            model, df_model = train_and_save_model(df)
        st.success("Model berhasil dilatih dan disimpan!")

    # ========================
    #  DASHBOARD
    # ========================
    if page == "Dashboard":
        st.subheader("📊 Dashboard Penjualan")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Penjualan", f"{df['Produk_Terjual'].sum():,.0f} unit")
        col2.metric("Rata-rata Harian", f"{df['Produk_Terjual'].mean():,.1f} unit")
        col3.metric("Total Pembeli", f"{df['Pembeli'].sum():,.0f} orang")

        st.markdown("### Grafik Penjualan")
        fig = plot_sales_data_daily(df)
        st.pyplot(fig)

        st.markdown("### Data Penjualan Terbaru")
        df_display = df.tail(10).sort_values('Tanggal_Transaksi', ascending=False)
        df_display['Tanggal_Transaksi'] = df_display['Tanggal_Transaksi'].dt.strftime('%Y-%m-%d')
        st.dataframe(df_display)

    # ========================
    #  PREDIKSI STOK
    # ========================
    elif page == "Prediksi Stok":
        st.subheader("📈 Prediksi Kebutuhan Stok")
        n_hari = st.slider("Jumlah hari ke depan", 7, 180, 30)
        safety_pct = st.slider("Safety stock (%)", 0, 50, 20)

        if st.button("Prediksi"):
            hasil, total_stok = predict_stock_daily(model, df_model, n_hari, safety_pct)
            col1, col2 = st.columns(2)
            col1.metric("Total Stok Disarankan", f"{total_stok:,.0f} unit")
            col2.metric("Rata-rata Penjualan/Hari", f"{hasil['Prediksi_Produk_Terjual'].mean():,.1f} unit")

            st.markdown("### Grafik Prediksi")
            fig = plot_sales_data_daily(df, hasil)
            st.pyplot(fig)

            hasil_display = hasil.copy()
            hasil_display['Tanggal'] = hasil_display['Tanggal'].dt.strftime('%Y-%m-%d')
            st.markdown("### Detail Prediksi Harian")
            st.dataframe(hasil_display.set_index('Tanggal'))

    # ========================
    #  ANALISIS PENJUALAN
    # ========================
    elif page == "Analisis Penjualan":
        st.subheader("📊 Analisis Penjualan")
        st.write("Bagian ini dapat digunakan untuk menganalisis tren penjualan berdasarkan variabel seperti pembeli, tayangan halaman, dan pesanan.")
        corr = df[["Produk_Terjual","Pembeli","Tayangan_Halaman","Kunjungan_Halaman_Toko","Pesanan"]].corr()
        st.markdown("### Korelasi Antar Variabel")
        st.dataframe(corr)

        st.markdown("### Heatmap Korelasi")
        plt.figure(figsize=(8,6))
        sns.heatmap(corr, annot=True, cmap="Blues")
        st.pyplot(plt)

    # ========================
    #  TENTANG
    # ========================
    elif page == "Tentang":
        st.subheader("ℹ️ Tentang Aplikasi")
        st.write("""
        Aplikasi ini dikembangkan oleh **Xila Studio** sebagai sistem pendukung keputusan
        untuk memprediksi kebutuhan stok hoodie berdasarkan tren penjualan historis.
        \n
        📌 Fitur:
        - Dashboard penjualan interaktif  
        - Prediksi stok harian & bulanan  
        - Analisis korelasi penjualan  
        - Tampilan profesional dengan animasi UI
        \n
        🧠 Model: Linear Regression  
        📊 Data: File `dataset.csv` historis
        """)

# Jalankan aplikasi
if __name__ == "__main__":
    main()
