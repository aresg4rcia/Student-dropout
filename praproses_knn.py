# =====================================================================
# TUGAS P2 - Representasi Data & Pra-pemrosesan (Pembelajaran Mesin)
# Dataset  : Predict Students' Dropout and Academic Success (UCI, id 697)
# Link     : https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
# Algoritma: K-Nearest Neighbors (KNN)
# =====================================================================
# Cara pakai: taruh file data.csv di folder yang sama dengan program ini
# (di Google Colab: upload data.csv lalu jalankan semua sel).

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CSV_PATH = "data.csv"
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)


def judul(teks):
    print("\n" + "=" * 70 + "\n" + teks + "\n" + "=" * 70)


# ---------------------------------------------------------------------
# 1. MEMBACA DATA & INFO KOLOM
# File memakai pemisah titik koma (;) dan diawali karakter BOM,
# sehingga dibaca dengan sep=';' dan encoding='utf-8-sig'.
# ---------------------------------------------------------------------
judul("1. INFO DATA & TIPE KOLOM (SEBELUM)")
df = pd.read_csv(CSV_PATH, sep=";", encoding="utf-8-sig")
print("Ukuran data (baris, kolom):", df.shape)
print(df.dtypes.value_counts())

# Rapikan nama kolom: ada karakter tab di akhir 'Daytime/evening attendance'
# dan salah eja 'Nacionality'.
df.columns = df.columns.str.strip()
df = df.rename(columns={"Nacionality": "Nationality"})

# ---------------------------------------------------------------------
# 2. KOREKSI TIPE KOLOM
# Kolom nominal berkode angka (mis. Course 9500, 9147) bukan angka sebenarnya,
# jadi diubah menjadi kategori agar tidak dihitung sebagai jarak numerik.
# ---------------------------------------------------------------------
kolom_nominal = ["Marital status", "Application mode", "Course",
                 "Previous qualification", "Mother's qualification",
                 "Father's qualification"]
for k in kolom_nominal:
    df[k] = df[k].astype("category")

judul("2. TIPE KOLOM SETELAH DIKOREKSI")
print(df.dtypes.value_counts())

# ---------------------------------------------------------------------
# 3. MISSING VALUE & DUPLIKASI
# ---------------------------------------------------------------------
judul("3. MISSING VALUE & DUPLIKASI")
print("Total missing value :", int(df.isna().sum().sum()))
print("Jumlah baris duplikat:", int(df.duplicated().sum()))
df = df.drop_duplicates()          # tidak ada yang terhapus, tetap dijalankan
df = df.dropna()                   # tidak ada yang terhapus, tetap dijalankan

# Nilai 0 pada nilai semester: bukan missing value sebenarnya, tapi
# mahasiswa tanpa nilai lulus. Dibiarkan karena tetap informasi valid.
print("Nilai semester 1 = 0 :", int((df["Curricular units 1st sem (grade)"] == 0).sum()))
print("Nilai semester 2 = 0 :", int((df["Curricular units 2nd sem (grade)"] == 0).sum()))

# ---------------------------------------------------------------------
# 4. KATEGORI (CEK EJAAN / KAPITALISASI)
# Kolom teks hanya Target. Dirapikan spasi dan hurufnya.
# ---------------------------------------------------------------------
df["Target"] = df["Target"].astype(str).str.strip().str.title()
judul("4. KATEGORI TARGET & KETIDAKSEIMBANGAN KELAS")
print(df["Target"].value_counts())
print((df["Target"].value_counts(normalize=True) * 100).round(1).astype(str) + " %")

# ---------------------------------------------------------------------
# 5. INDIKASI KEBOCORAN DATA (LEAKAGE) - cek sederhana
# Korelasi fitur numerik dengan kelas Graduate (satu lawan sisanya).
# ---------------------------------------------------------------------
judul("5. CEK KEBOCORAN DATA (korelasi tertinggi dengan Graduate)")
is_grad = (df["Target"] == "Graduate").astype(int)
kor = df.select_dtypes("number").corrwith(is_grad).abs().sort_values(ascending=False)
print(kor.head(5).round(3))
print("Tidak ada fitur yang berkorelasi mendekati 1, jadi tidak ada bocoran langsung.")
print("Catatan: fitur semester 2 baru tersedia di akhir tahun pertama.")

# ---------------------------------------------------------------------
# 6. SELEKSI FITUR (BUANG FITUR TIDAK RELEVAN)
# - Nationality: 97,5% bernilai sama dan redundan dengan International
# - Mother's/Father's occupation: ~190 kode, terlalu banyak untuk KNN
# ---------------------------------------------------------------------
df = df.drop(columns=["Nationality", "Mother's occupation", "Father's occupation"])
judul("6. SETELAH SELEKSI FITUR")
print("Jumlah fitur sekarang:", df.shape[1] - 1)

# ---------------------------------------------------------------------
# 7. PEMBAGIAN DATA (80% latih : 20% uji, stratified)
# Dilakukan SEBELUM outlier, encoding, dan scaling supaya tidak ada
# informasi data uji yang bocor ke data latih.
# ---------------------------------------------------------------------
X = df.drop(columns="Target")
y = df["Target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
judul("7. PEMBAGIAN DATA")
print("Data latih:", X_train.shape, "| Data uji:", X_test.shape)

# ---------------------------------------------------------------------
# 8. OUTLIER (IQR) PADA 'Age at enrollment'
# Batas dihitung dari data latih, nilai ekstrem dibatasi (winsorizing)
# karena usia 40-70 tahun valid tapi ekstrem.
# ---------------------------------------------------------------------
kol = "Age at enrollment"
q1, q3 = X_train[kol].quantile([0.25, 0.75])
iqr = q3 - q1
batas_bawah, batas_atas = q1 - 1.5 * iqr, q3 + 1.5 * iqr
judul("8. OUTLIER IQR - " + kol)
print(f"Q1={q1}, Q3={q3}, IQR={iqr}, batas=[{batas_bawah}, {batas_atas}]")
print("Outlier data latih:", int(((X_train[kol] < batas_bawah) | (X_train[kol] > batas_atas)).sum()))
X_train[kol] = X_train[kol].clip(batas_bawah, batas_atas)
X_test[kol] = X_test[kol].clip(batas_bawah, batas_atas)

# ---------------------------------------------------------------------
# 9. ENCODING KATEGORI (ONE-HOT)
# Kategori yang jarang (<1% data latih) digabung menjadi 'Lainnya' agar
# jumlah kolom tidak membengkak (KNN sensitif pada banyak dimensi).
# ---------------------------------------------------------------------
for k in kolom_nominal:
    frek = X_train[k].astype(str).value_counts(normalize=True)
    dipakai = frek[frek >= 0.01].index
    for d in (X_train, X_test):
        d[k] = d[k].astype(str).where(d[k].astype(str).isin(dipakai), "Lainnya")

X_train = pd.get_dummies(X_train, columns=kolom_nominal, dtype=int)
X_test = pd.get_dummies(X_test, columns=kolom_nominal, dtype=int)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
judul("9. ENCODING ONE-HOT")
print("Jumlah kolom setelah encoding:", X_train.shape[1])

# ---------------------------------------------------------------------
# 10. STANDARISASI FITUR NUMERIK (StandardScaler)
# KNN berbasis jarak, jadi fitur berskala besar (nilai 0-200) akan
# mendominasi fitur berskala kecil (0-1). Scaler hanya di-fit pada data latih.
# Kolom biner/one-hot (0/1) tidak di-scale.
# ---------------------------------------------------------------------
kolom_numerik = ["Application order", "Previous qualification (grade)", "Admission grade",
                 "Age at enrollment", "Unemployment rate", "Inflation rate", "GDP"] + \
                [c for c in X_train.columns if c.startswith("Curricular units")]
scaler = StandardScaler()
X_train[kolom_numerik] = scaler.fit_transform(X_train[kolom_numerik])
X_test[kolom_numerik] = scaler.transform(X_test[kolom_numerik])

# ---------------------------------------------------------------------
# 11. HASIL AKHIR: 5 DATA TERATAS & 5 DATA TERBAWAH
# ---------------------------------------------------------------------
hasil_train = X_train.assign(Target=y_train.values)
hasil_test = X_test.assign(Target=y_test.values)
# Gabungkan data latih dan uji, urutkan kembali sesuai urutan data asli
hasil_semua = pd.concat([hasil_train, hasil_test]).sort_index()

# Karena kolom hasil one-hot sangat banyak (80 kolom), yang ditampilkan
# adalah kolom-kolom penting saja. Data lengkap tersimpan di file CSV.
kolom_tampil = ["Application order", "Admission grade", "Age at enrollment",
                "Debtor", "Tuition fees up to date", "Gender", "Scholarship holder",
                "Curricular units 1st sem (grade)", "Curricular units 2nd sem (grade)",
                "Marital status_1", "Course_9500", "Target"]
judul("11. HASIL PRAPROSES - SELURUH DATA")
print("Ukuran lengkap (baris, kolom):", hasil_semua.shape)
print("\n5 DATA TERATAS:")
print(hasil_semua[kolom_tampil].head(5))
print("\n5 DATA TERBAWAH:")
print(hasil_semua[kolom_tampil].tail(5))

hasil_semua.to_csv("data_hasil_praproses.csv", index=False)
print("\nTersimpan: data_hasil_praproses.csv (data lengkap, semua kolom)")
