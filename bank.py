import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('bank.csv', sep=";")
    return df

df = load_data()

# Display basic info and first rows
st.title("Interactive Bank Marketing Dataset Exploration")
st.write("Dataset ini berisi data kampanye pemasaran telepon oleh bank Portugis.")
st.write("Kami akan mengeksplorasi bagaimana variabel-variabel dalam dataset ini mempengaruhi hasil pemasaran.")

# Display first few rows
st.subheader("First few rows of the dataset")
st.dataframe(df.head())

# Sidebar for user interaction
st.sidebar.title("Filter Data")

# Filter by job
job_filter = st.sidebar.multiselect("Filter by Job", df['job'].unique(), df['job'].unique())
filtered_data = df[df['job'].isin(job_filter)]

# Filter by marital status
marital_filter = st.sidebar.multiselect("Filter by Marital Status", df['marital'].unique(), df['marital'].unique())
filtered_data = filtered_data[filtered_data['marital'].isin(marital_filter)]

# Show filtered data
st.subheader(f"Data filtered by Job and Marital Status")
st.dataframe(filtered_data)

# Show summary statistics of numeric columns
st.subheader("Summary Statistics of Numeric Columns")
st.write(filtered_data.describe())

# Visualizations

# Histogram for age
st.subheader("Age Distribution")
fig, ax = plt.subplots()
sns.histplot(filtered_data['age'], kde=True, ax=ax)
st.pyplot(fig=fig)  # Pass the 'fig' explicitly

# Countplot for job category
st.subheader("Job Distribution")
fig, ax = plt.subplots()
sns.countplot(x='job', data=filtered_data, ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig=fig)  # Pass the 'fig' explicitly

# Scatter plot for duration vs. age, colored by target variable (y)
st.subheader("Duration vs Age colored by Subscription Outcome")
fig, ax = plt.subplots()
sns.scatterplot(data=filtered_data, x='age', y='duration', hue='y', palette='coolwarm', ax=ax)
st.pyplot(fig=fig)  # Pass the 'fig' explicitly

# Pairplot for numeric features
st.subheader("Pairplot for Numeric Features")
sns.pairplot(filtered_data[['age', 'balance', 'duration', 'campaign', 'previous', 'pdays', 'y']], hue='y', palette='coolwarm')
st.pyplot(fig=fig)  # Pass the 'fig' explicitly

# Train a RandomForest model and show accuracy score (simple classifier for exploration)
X = filtered_data[['age', 'balance', 'duration', 'campaign', 'previous', 'pdays']]
y = LabelEncoder().fit_transform(filtered_data['y'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Show accuracy
accuracy = accuracy_score(y_test, y_pred)
st.subheader(f"Random Forest Classifier Accuracy: {accuracy:.2f}")

# Conclusion
st.write("Dari visualisasi dan model yang disajikan, Anda dapat mengeksplorasi faktor-faktor yang mempengaruhi keputusan pelanggan untuk berlangganan deposito berjangka.")
