import streamlit as st
import pandas as pd
import plotly.express as px
import yfinance as yf

# Fungsi untuk mengambil data saham
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="5y")  # Data historis 5 tahun
    data['Date'] = data.index
    return data[['Date', 'Close']]

# Ambil data saham untuk Netflix, Amazon, dan Meta
netflix_data = get_stock_data('NFLX')
amazon_data = get_stock_data('AMZN')
meta_data = get_stock_data('META')

# Gabungkan data menjadi satu DataFrame untuk kemudahan perbandingan
df = pd.concat([netflix_data.set_index('Date').rename(columns={'Close': 'Netflix'}),
                amazon_data.set_index('Date').rename(columns={'Close': 'Amazon'}),
                meta_data.set_index('Date').rename(columns={'Close': 'Meta'})], axis=1)

# Layout dan judul Streamlit
st.title('Fluktuasi Harga Saham: Netflix, Amazon, Meta')

# Sidebar untuk memilih rentang tanggal
st.sidebar.header("Rentang Tanggal")
start_date = st.sidebar.date_input("Tanggal Mulai", df.index.min())
end_date = st.sidebar.date_input("Tanggal Selesai", df.index.max())

# Filter data berdasarkan rentang tanggal yang dipilih
filtered_df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]

# Membuat plot
fig = px.line(filtered_df, x=filtered_df.index, y=['Netflix', 'Amazon', 'Meta'],
              labels={'x': 'Tanggal', 'value': 'Harga Saham'},
              title="Perbandingan Harga Saham (Netflix, Amazon, Meta)")

# Menampilkan plot
st.plotly_chart(fig)
