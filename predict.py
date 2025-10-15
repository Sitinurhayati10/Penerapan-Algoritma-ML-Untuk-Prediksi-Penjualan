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

# === Konfigurasi Tema Cantik & Responsif ===
st.set_page_config(
    page_title="Xila Studio - Sistem Prediksi Stok Hoodie", 
    layout="centered",  # Lebih responsif untuk HP
    initial_sidebar_state="collapsed"  # Lebih rapi di mobile
)

# CSS untuk tema cantik, modern, dan responsif
st.markdown("""
<style>
    :root {
        --primary: #6C63FF;
        --secondary: #4A90E2;
        --accent: #FF6584;
        --success: #4CAF50;
        --light-bg: #FAFAFF;
        --card-bg: #FFFFFF;
        --text: #2D2D2D;
        --border: #E0E0F8;
    }

    .stApp {
        background-color: var(--light-bg);
        font-family: 'Segoe UI', 'Roboto', sans-serif;
    }

    /* Responsif untuk HP */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 1.8rem !important;
        }
        .main-header h2 {
            font-size: 1.3rem !important;
        }
        .stMetric {
            padding: 10px !important;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.2rem !important;
        }
        .element-container img {
            width: 80px !important;
            height: auto !important;
        }
    }

    /* Animasi halus */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .fade-in {
        animation: fadeInUp 0.8s ease-out;
    }

    /* Header */
    .main-header {
        text-align: center;
        margin: 10px 0 20px;
        padding: 10px;
        border-radius: 12px;
        background: linear-gradient(135deg, #f5f7ff, #eef1ff);
        box-shadow: 0 4px 12px rgba(108, 99, 255, 0.1);
    }

    .main-header h1 {
        color: var(--primary);
        font-weight: 700;
        margin-bottom: 5px;
    }

    .main-header h2 {
        color: var(--text);
        font-weight: 500;
        opacity: 0.9;
    }

    /* Card & Metric */
    .stMetric {
        background: var(--card-bg);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        border-top: 3px solid var(--primary);
        transition: transform 0.2s, box-shadow 0.2s;
    }

    .stMetric:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.1);
    }

    div[data-testid="stMetricValue"] {
        color: var(--primary);
        font-size: 1.5rem;
        font-weight: 700;
    }

    div[data-testid="stMetricLabel"] {
        color: var(--text);
        font-size: 0.95rem;
        font-weight: 600;
    }

    /* Tombol */
    .stButton > button {
        background: var(--primary);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 4px 8px rgba(108, 99, 255, 0.3);
        transition: all 0.3s ease;
        width: 100%;
    }

    .stButton > button:hover {
        background: #5a52d5;
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(108, 99, 255, 0.4);
    }

    /* Sidebar (jika dibuka di HP) */
    .stSidebar {
        background: linear-gradient(to bottom, #6C63FF, #5A52D5);
        color: white;
    }

    .stSidebar .stSelectbox > div > div > div {
        color: white !important;
    }

    .stSidebar label {
        color: white !important;
    }

    /* Tabel & Grafik */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }

    /* Divider */
    hr {
        border-color: var(--border);
        margin: 20px 0;
    }

    /* Info box */
    .stAlert {
        background: #f0f4ff;
        border-left: 4px solid var(--primary);
        border-radius: 8px;
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
    
    future_months = []
    current_date = last_date + pd.DateOffset(months=1)
    for i in range(n_bulan):
        month_start = current_date.replace(day=1)
        month_end = (month_start + pd.DateOffset(months=1)) - pd.DateOffset(days=1)
        future_months.append((month_start, month_end))
        current_date = month_end + pd.DateOffset(days=1)
    
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
    
    future_df["Days_in_Month"] = future_df.apply(
        lambda row: (row["Month_End"] - row["Month_Start"]).days + 1, axis=1
    )
    
    X_future = future_df[[
        "tren_hari", "bulan",
        "Pembeli", "Tayangan_Halaman", "Kunjungan_Halaman_Toko", "Pesanan"
    ]]
    
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
    plt.figure(figsize=(10, 5))
    plt.plot(df['Tanggal_Transaksi'], df['Produk_Terjual'], 
             label='Data Historis', color='#6C63FF', linewidth=2.2)
    if hasil_prediksi is not None:
        plt.plot(hasil_prediksi['Tanggal'], hasil_prediksi['Prediksi_Produk_Terjual'], 
                 label='Prediksi', color='#FF6584', linewidth=2.2, linestyle='--')
    plt.title('Grafik Penjualan Harian', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Tanggal', fontsize=11)
    plt.ylabel('Jumlah Terjual', fontsize=11)
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout()
    return plt

# === Fungsi: Visualisasi Data Bulanan ===
def plot_sales_data_monthly(df, hasil_prediksi=None):
    df_bulanan = df.copy()
    df_bulanan['Bulan'] = df_bulanan['Tanggal_Transaksi'].dt.to_period('M')
    df_bulanan = df_bulanan.groupby('Bulan')['Produk_Terjual'].sum().reset_index()
    df_bulanan['Bulan'] = df_bulanan['Bulan'].astype(str)
    
    plt.figure(figsize=(10, 5))
    plt.bar(df_bulanan['Bulan'], df_bulanan['Produk_Terjual'], 
             label='Data Historis', color='#6C63FF', alpha=0.8)
    if hasil_prediksi is not None:
        plt.bar(hasil_prediksi['Bulan'], hasil_prediksi['Prediksi_Produk_Terjual'], 
                 label='Prediksi', color='#FF6584', alpha=0.8)
    plt.title('Grafik Penjualan Bulanan', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Bulan', fontsize=11)
    plt.ylabel('Jumlah Terjual', fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.2)
    plt.legend()
    plt.tight_layout()
    return plt

# === Aplikasi Utama ===
def main():
    # Header
    st.markdown("""
    <div class="main-header fade-in">
        <h1>Xila Studio</h1>
        <h2>Sistem Prediksi Stok Hoodie</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Cek data
    if not Path(DATA_PATH).exists():
        st.error("File 'dataset.csv' tidak ditemukan di folder ini!")
        st.stop()
    
    df = load_and_clean_data(DATA_PATH)
    if df is None:
        st.stop()
    
    # Muat atau latih model
    if Path(MODEL_PATH).exists():
        model = joblib.load(MODEL_PATH)
        df_model = df.copy()
        df_model["hari_dalam_minggu"] = df_model["Tanggal_Transaksi"].dt.dayofweek
        df_model["bulan"] = df_model["Tanggal_Transaksi"].dt.month
    else:
        with st.spinner("Melatih model..."):
            model, df_model = train_and_save_model(df)
        st.success("✅ Model berhasil dilatih dan disimpan!")

    # Navigasi sederhana (lebih ramah HP)
    page = st.selectbox("🧭 Pilih Halaman", ["Dashboard", "Prediksi Stok", "Analisis Penjualan", "Tentang"])

    if page == "Dashboard":
        st.markdown("### 📊 Dashboard Penjualan")
        
        # Metrik (responsif: 2 kolom di HP, 4 di desktop)
        cols = st.columns([1, 1] if st.session_state.get('mobile', False) else [1, 1, 1, 1])
        metrics = [
            ("Total Penjualan", f"{df['Produk_Terjual'].sum():,.0f} unit"),
            ("Rata-rata Harian", f"{df['Produk_Terjual'].mean():,.1f} unit"),
            ("Total Pembeli", f"{df['Pembeli'].sum():,.0f} orang"),
        ]
        konversi_values = pd.to_numeric(df['Persentase_Konversi'], errors='coerce')
        konversi_avg = f"{konversi_values.mean():.2f}%" if konversi_values.notna().any() else "N/A"
        metrics.append(("Konversi Rata-rata", konversi_avg))
        
        for i, (label, value) in enumerate(metrics):
            with cols[i % len(cols)]:
                st.metric(label, value)
        
        st.markdown("### 📈 Tren Penjualan")
        fig = plot_sales_data_daily(df)
        st.pyplot(fig)
        
        st.markdown("### 📋 Data Terbaru")
        df_display = df.tail(5).sort_values('Tanggal_Transaksi', ascending=False).copy()
        df_display['Tanggal_Transaksi'] = df_display['Tanggal_Transaksi'].dt.strftime('%Y-%m-%d')
        st.dataframe(df_display, use_container_width=True)

    elif page == "Prediksi Stok":
        st.markdown("### 🔮 Prediksi Kebutuhan Stok")
        prediksi_type = st.radio("Jenis Prediksi", ["Harian", "Bulanan"], horizontal=True)
        
        if prediksi_type == "Harian":
            col1, col2 = st.columns(2)
            with col1:
                n_hari = st.slider("Jumlah hari ke depan", 7, 180, 30)
            with col2:
                safety_pct = st.slider("Safety stock (%)", 0, 50, 20)
            
            if st.button("✨ Prediksi Stok Harian"):
                with st.spinner("Memproses..."):
                    hasil, total_stok = predict_stock_daily(model, df_model, n_hari, safety_pct)
                
                st.markdown(f"### 📅 Hasil Prediksi ({n_hari} Hari)")
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Stok", f"{total_stok:,.0f}")
                col2.metric("Rata-rata/Hari", f"{hasil['Prediksi_Produk_Terjual'].mean():,.1f}")
                col3.metric("Safety Stock", f"{total_stok - hasil['Prediksi_Produk_Terjual'].mean() * n_hari:,.1f}")
                
                st.pyplot(plot_sales_data_daily(df, hasil))
                hasil['Tanggal'] = hasil['Tanggal'].dt.strftime('%Y-%m-%d')
                st.dataframe(hasil.set_index('Tanggal'), use_container_width=True)
                st.info(f"📌 Rekomendasi: Pesan **{total_stok:,.0f} unit** untuk {n_hari} hari ke depan.")

        else:
            col1, col2 = st.columns(2)
            with col1:
                n_bulan = st.slider("Jumlah bulan ke depan", 1, 12, 3)
            with col2:
                safety_pct = st.slider("Safety stock (%)", 0, 50, 20)
            
            if st.button("✨ Prediksi Stok Bulanan"):
                with st.spinner("Memproses..."):
                    hasil, total_stok = predict_stock_monthly(model, df_model, n_bulan, safety_pct)
                
                st.markdown(f"### 📅 Hasil Prediksi ({n_bulan} Bulan)")
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Stok", f"{total_stok:,.0f}")
                col2.metric("Rata-rata/Bulan", f"{hasil['Prediksi_Produk_Terjual'].mean():,.1f}")
                col3.metric("Safety Stock", f"{total_stok - hasil['Prediksi_Produk_Terjual'].sum():,.1f}")
                
                st.pyplot(plot_sales_data_monthly(df, hasil))
                st.dataframe(hasil.set_index('Bulan'), use_container_width=True)
                st.info(f"📌 Rekomendasi: Pesan **{total_stok:,.0f} unit** untuk {n_bulan} bulan ke depan.")

    elif page == "Analisis Penjualan":
        st.markdown("### 🔍 Analisis Penjualan")
        analisis_type = st.selectbox("Jenis Analisis", ["Penjualan per Bulan", "Penjualan per Hari"])
        
        if analisis_type == "Penjualan per Bulan":
            df_bulanan = df.copy()
            df_bulanan['Bulan'] = df_bulanan['Tanggal_Transaksi'].dt.to_period('M')
            df_bulanan = df_bulanan.groupby('Bulan').agg({'Produk_Terjual': 'sum'}).reset_index()
            df_bulanan['Bulan'] = df_bulanan['Bulan'].astype(str)
            st.dataframe(df_bulanan, use_container_width=True)
            st.pyplot(plot_sales_data_monthly(df))
        else:
            df['Hari'] = df['Tanggal_Transaksi'].dt.day_name()
            hari_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            df_harian = df.groupby('Hari')['Produk_Terjual'].mean().reindex(hari_order).reset_index()
            st.dataframe(df_harian, use_container_width=True)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(df_harian['Hari'], df_harian['Produk_Terjual'], color='#FF6584')
            plt.title('Rata-rata Penjualan per Hari', fontweight='bold')
            plt.xticks(rotation=45)
            st.pyplot(fig)

    elif page == "Tentang":
        st.markdown("### 🌟 Tentang Xila Studio")
        st.markdown("""
        **Xila Studio** adalah toko online hoodie dengan desain unik dan kualitas premium.
        
        **Visi**: Menjadi merek hoodie pilihan anak muda Indonesia.
        
        **Sistem Ini**: Membantu prediksi stok agar tidak kehabisan barang dan mengurangi kelebihan stok.
        """)
        # Logo opsional — jika tidak ada, tidak error
        try:
            st.image("logo xila.jpg", width=200)
        except:
            st.info("Logo tidak ditemukan — aplikasi tetap berjalan.")

if __name__ == "__main__":
    main()
