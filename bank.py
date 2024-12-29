import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Membaca dataset
@st.cache
def load_data():
    data = pd.read_csv('bank.csv', sep=";")
    return data

# Memuat dataset
df = load_data()

# Encoder untuk kolom kategorikal
le = LabelEncoder()

# Kolom kategorikal yang perlu di-encode
cat_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']
for col in cat_columns:
    df[col] = le.fit_transform(df[col])

# Judul aplikasi
st.title("Dashboard Interaktif untuk Dataset Bank Marketing")
st.write("""
Dataset ini berhubungan dengan kampanye pemasaran langsung yang dilakukan oleh bank Portugal. 
Kampanye ini fokus pada panggilan telepon untuk menawarkan deposito bank. Tujuan kita adalah 
untuk menganalisis faktor-faktor yang mempengaruhi keputusan pelanggan dalam berlangganan deposito bank.
""")

# Menampilkan data
st.subheader("Data Set")
st.write(df.head())

# Statistik deskriptif
st.subheader("Statistik Deskriptif")
st.write(df.describe())

# Filter untuk memilih kolom numerik
st.subheader("Filter Data Berdasarkan Kolom")
selected_column = st.selectbox("Pilih kolom untuk analisis", df.columns)
st.write(df[selected_column].value_counts())

# Visualisasi distribusi usia
st.subheader("Distribusi Usia")
fig, ax = plt.subplots()
sns.histplot(df['age'], kde=True, ax=ax)
ax.set_title('Distribusi Usia Pelanggan')
st.pyplot(fig)

# Visualisasi proporsi pelanggan yang berlangganan deposito
st.subheader("Proporsi Pelanggan yang Berlangganan Deposito")
fig2, ax2 = plt.subplots()
sns.countplot(x='y', data=df, ax=ax2)
ax2.set_title('Distribusi Berlangganan Deposito')
st.pyplot(fig2)

# Visualisasi hubungan durasi dan berlangganan deposito
st.subheader("Hubungan Durasi Kontak dan Berlangganan Deposito")
fig3, ax3 = plt.subplots()
sns.boxplot(x='y', y='duration', data=df, ax=ax3)
ax3.set_title('Durasi Kontak vs Berlangganan Deposito')
st.pyplot(fig3)

# Visualisasi job vs target (y)
st.subheader("Hubungan Jenis Pekerjaan dan Berlangganan Deposito")
fig4, ax4 = plt.subplots(figsize=(10,6))
sns.countplot(x='job', hue='y', data=df, ax=ax4)
ax4.set_title('Jenis Pekerjaan vs Berlangganan Deposito')
st.pyplot(fig4)
