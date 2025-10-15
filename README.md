# Prediksi Stok — E-Commerce

File singkat untuk menjalankan prediksi stok menggunakan model yang sudah disimpan (`stok_forecast_model.joblib`) dan data CSV (`dataset.csv`).

Persyaratan:

- Python 3.10+ (disarankan)
- Virtual environment (opsional tetapi disarankan)
- Paket di `Requirements.txt.txt` (streamlit, pandas, numpy, scikit-learn, joblib,...)

Cara singkat (PowerShell):

```powershell
# Buat dan aktifkan virtualenv (opsional)
python -m venv .venv; .\.venv\Scripts\Activate.ps1

# Instal requirement
pip install -r .\Requirements.txt.txt

# Jalankan prediksi (contoh 30 hari, safety 20%)
python .\predict.py -d 30 -s 20 -o predictions.csv
```

Output:

- `predictions.csv` — prediksi penjualan harian untuk jumlah hari yang diminta
- Konsol akan menampilkan ringkasan singkat (rata-rata per hari, total stok rekomendasi)

Catatan:

- Script `predict.py` meniru preprocessing di `main.py`.
- Jika `stok_forecast_model.joblib` tidak ada, jalankan `main.py` (Streamlit) untuk melatih dan menyimpan model, atau salin file joblib yang valid ke folder ini.
