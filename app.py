
import streamlit as st
import pandas as pd
import joblib

# Load the dataset
df = pd.read_csv('financial_risk_assessment.csv')  # Ensure the dataset is in the working directory

st.markdown("<h1 style='text-align: center; font-size: 50px; color :Blue;'>Risk Rating Prediction</h1>", unsafe_allow_html=True)
# Drop rows where 'Risk Rating' is 'Medium'
df_filtered = df[df['Risk Rating'] != 'Medium']

# Function to assign city value based on the Risk Rating
def get_city_value(city_name):
    city_data = df_filtered[df_filtered['City'] == city_name]
    if (city_data['Risk Rating'] == 'High').all():
        return 1
    else:
        return 0

# Get unique cities for the dropdown
unique_cities = df_filtered['City'].unique()

# Load the trained model
model = joblib.load('best_model.pkl')  # Replace with your model file path

# Streamlit UI setup with yellow border and white background
st.markdown("""
    <style>
        body {
            background-color: white;
            color: black;
            border: 5px solid yellow; /* Yellow border */
            padding: 20px; /* Add some padding for spacing */
        }
        .stSlider > div > div > div {
            font-size: 20px; /* Increase font size for sliders */
        }
        .stSelectbox > div > label {
            font-size: 50px; /* Increase label font size */
        }
        .stNumberInput > div > label {
            font-size: 50px; /* Increase label font size */
        }
        .stButton > button {
            font-size: 20px; /* Increase font size for buttons */
        }
    </style>
    """, unsafe_allow_html=True)


# Gender selection
st.markdown("<br><h5 style='color:  #ec085e ;font-size: 30px;'>Gender</h5>", unsafe_allow_html=True)
gender = st.selectbox(' ', ['Male', 'Female'], help="Select your gender.")
gender_female = 1 if gender == 'Female' else 0
gender_male = 1 if gender == 'Male' else 0

st.markdown("<h5 style='color:  #ec085e ;font-size: 30px;'>Age</h5>", unsafe_allow_html=True)
age = st.slider(' ', 0, 100, 1, help="Select Age.")

# City selection
st.markdown("<h5 style='color:  #ec085e ; font-size: 30px;'>City</h5>", unsafe_allow_html=True)
city = st.selectbox(' ', options=unique_cities, format_func=lambda x: x, help="Select your city.")
city_value = get_city_value(city)  # Get the city value based on Risk Rating (High = 1, Low = 0)

# Adding styled labels for Payment History as a dropdown
st.markdown("<h5 style='color:  #ec085e ;font-size: 30px;'>Payment History</h5>", unsafe_allow_html=True)
payment_history_options = ['Bad', 'Poor', 'Good', 'Excellent']
payment_history_scores = [0, 1, 2, 3]
payment_history = st.selectbox(' ', options=payment_history_options, format_func=lambda x: x, help="Select your payment history rating.")
payment_history_score = payment_history_scores[payment_history_options.index(payment_history)]



st.markdown("<br><h5 style='color:  #ec085e ;font-size: 30px;'>Marital Status</h5>", unsafe_allow_html=True)
marital_status = st.selectbox(' ', ['Single', 'Married'], help="Select your marital status.")
marital_status_married = 1 if marital_status == 'Married' else 0
marital_status_single = 1 if marital_status == 'Single' else 0


st.markdown("<br><h5 style='color:  #ec085e ;font-size: 30px;'>Loan Purpose</h5>", unsafe_allow_html=True)
loan_purpose = st.selectbox(' ', ['Auto', 'Business', 'Personal'], help="Select the purpose of the loan.")
loan_purpose_auto = 1 if loan_purpose == 'Auto' else 0
loan_purpose_business = 1 if loan_purpose == 'Business' else 0
loan_purpose_personal = 1 if loan_purpose == 'Personal' else 0

st.markdown("<h5 style='color:  #ec085e ;font-size: 30px;'>Loan Amount</h5>", unsafe_allow_html=True)
loan_amount = st.number_input(' ', value=10010, help="Enter the amount of the loan.")

st.markdown("<h5 style='color:  #ec085e ;font-size: 30px;'>Income</h5>", unsafe_allow_html=True)
assets_value = st.number_input(' ', value=50000, help="Enter the total value of your assets.")


st.markdown("<br><h5 style='color:  #ec085e ;font-size: 30px;'>Employment Status</h5>", unsafe_allow_html=True)
employment_status = st.selectbox(' ', ['Employed', 'Self-employed', 'Unemployed'], help="Select your employment status.")
employment_status_employed = 1 if employment_status == 'Employed' else 0
employment_status_self_employed = 1 if employment_status == 'Self-employed' else 0
employment_status_unemployed = 1 if employment_status == 'Unemployed' else 0

st.markdown("<h5 style='color:  #ec085e ;font-size: 30px;'>Years at Current Job</h5>", unsafe_allow_html=True)
years_at_current_job = st.slider(' ', 0, 40, 5, help="Select the number of years you've been at your current job.")

# Prepare input data for prediction
input_data = {
    'Payment History': payment_history_score,  # Use the numerical score
    'City': city_value,
    'Marital Status Change': 1,
    'Gender_Female': gender_female,
    'Gender_Male': gender_male,
    'Marital Status_Married': marital_status_married,
    'Marital Status_Single': marital_status_single,
    'Loan Purpose_Auto': loan_purpose_auto,
    'Loan Purpose_Business': loan_purpose_business,
    'Loan Purpose_Personal': loan_purpose_personal,
    'Employment Status_Employed': employment_status_employed,
    'Employment Status_Unemployed': employment_status_unemployed,
    'Loan Amount': loan_amount,
    'Assets Value': assets_value,
    'Years at Current Job': years_at_current_job
}

# Convert input data to a DataFrame
input_df = pd.DataFrame([input_data])

# Ensure the input matches the model's expected columns
for col in model.feature_names_in_:
    if col not in input_df.columns:
        input_df[col] = 0  # Add missing columns
st.markdown("<br>", unsafe_allow_html=True)

# Predict button
if st.button('Predict Risk Rating'):
    # Reorder columns to match the model's training data
    input_df = input_df[model.feature_names_in_]
    
    # Make prediction
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0]

    # Display the result
    if prediction[0] == 1:  # High Risk
        st.markdown(f"<h3 style='color: red;text-align: center;'>Our Prediction is you are in: High Risk</h3>", unsafe_allow_html=True)

    else:  # Low Risk
        st.markdown(f"<h3 style='color: green;text-align: center;'>Our Prediction is you are in: Low Risk</h3>", unsafe_allow_html=True)
      
st.markdown("""
    <style>
        div.stButton {
            text-align: center;
            font-size: 30px; 
        }
        .stButton > button {
            font-size: 30px;  /* Increase font size */
            background-color: #f2511b;  /* Change background color to orange */
            color: white;  /* Change text color to white */
            padding: 18px 25px;  /* Add padding to increase button size */
            border: none;  /* Remove border */
            border-radius: 5px;  /* Add border radius for rounded corners */
            cursor: pointer;  /* Change cursor to pointer on hover */
        }
        .stButton > button:hover {
            background-color: darkorange;  /* Change color on hover */
        }
    </style>
""", unsafe_allow_html=True)
