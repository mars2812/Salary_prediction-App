import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
import streamlit as st
import joblib
import matplotlib.pyplot as plt

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv('salary_prediction.csv')

data = load_data()

# Handle missing values
imputer = SimpleImputer(strategy='mean')
data[['Age', 'Years of Experience']] = imputer.fit_transform(data[['Age', 'Years of Experience']])

# Feature engineering
data['gender_encoded'] = data['Gender'].map({'Male': 0, 'Female': 1})
data['education_encoded'] = data['Education Level'].map({'High School': 0, 'Bachelors': 1, 'Masters': 2, 'PhD': 3})

# Drop rows with missing encoded values if any
data.dropna(subset=['gender_encoded', 'education_encoded'], inplace=True)

# Define feature set and target variable
X = data[['Age', 'gender_encoded', 'education_encoded', 'Years of Experience']]
y = data['Salary']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'salary_predictor_model.pkl')

# Load the trained model
model = joblib.load('salary_predictor_model.pkl')

# Prediction and evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
accuracy_percentage = r2 * 100

# Streamlit App
st.markdown("""
    <div style="background-color: #ADD8E6; padding: 10px; border-radius: 5px;">
        <h1 style="color: black; text-align: center;"><b>Resume Salary Predictor</b></h1>
    </div>
""", unsafe_allow_html=True)

# Description in bold with colorful appearance
st.markdown("""
    <p style="font-size:18px;  color: #FF5733;">
        Predict your expected annual salary based on your resume details
        <br>Provide the necessary information in the sidebar, and the model will predict your expected annual salary package.
    </p>
""", unsafe_allow_html=True)

# Sidebar for user input
st.sidebar.header('User Input Parameters')
age = st.sidebar.number_input('Age', min_value=18, max_value=65, step=1)
gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
education_level = st.sidebar.selectbox('Highest Education Level', ['High School', 'Bachelors', 'Masters', 'PhD'])
experience_years = st.sidebar.number_input('Years of Experience', min_value=0, max_value=50, step=1)

# Encoding user input
gender_encoded = 0 if gender == 'Male' else 1
education_mapping = {'High School': 0, 'Bachelors': 1, 'Masters': 2, 'PhD': 3}
education_level_encoded = education_mapping[education_level]

if st.sidebar.button('Predict Salary'):
    try:
        user_data = np.array([[age, gender_encoded, education_level_encoded, experience_years]])
        predicted_annual_salary = model.predict(user_data)[0]
        predicted_monthly_salary = predicted_annual_salary / 12
        st.success(f'Expected Annual Salary Package: ${predicted_annual_salary:,.2f}')
        st.success(f'Expected Monthly Salary Package: ${predicted_monthly_salary:,.2f}')
    except Exception as e:
        st.error(f"Error: {e}")

# Show model evaluation
st.write(f'### Model Performance')
st.write(f'Mean Absolute Error on test set: ${mae:.2f}')
st.write(f'R² Score: {r2:.2f} ({accuracy_percentage:.2f}%)')

# Visualization
st.write('### Visualizations')

# Plot Actual vs Predicted Salaries
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.5)
ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--k', lw=2)
ax.set_xlabel('Actual Salary')
ax.set_ylabel('Predicted Salary')
ax.set_title('Actual vs Predicted Salaries')
st.pyplot(fig)

# Plot Salary Distribution
fig, ax = plt.subplots()
data['Salary'].hist(ax=ax, bins=20)
ax.set_xlabel('Salary')
ax.set_ylabel('Frequency')
ax.set_title('Salary Distribution')
st.pyplot(fig)

# Add some styling and footer
st.markdown("""
    <style>
        .reportview-container {
            background: #f9f9f9;
            padding: 20px;
        }
        .sidebar .sidebar-content {
            background: #e6e6e6;
        }
    </style>
    <footer style="text-align: center; padding: 10px; color: gray;">
        Developed by <span style="font-size:20px; font-weight:bold;">Matrika Dhamala</span>
    </footer>
""", unsafe_allow_html=True)
