import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model (Assume model.pkl exists after training)
def load_model():
    with open("./model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

# Function to predict charges
def predict_charges(age, sex, bmi, children, smoker, region):
    model = load_model()
    data = np.array([[age, sex, bmi, children, smoker, region]])
    prediction = model.predict(data)
    return prediction[0]

# Convert region to numerical values
def encode_region(region):
    region_mapping = {"Southwest": 0, "Southeast": 1, "Northwest": 2, "Northeast": 3}
    return region_mapping.get(region, -1)

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict", "History", "Model Statistics", "About"], key="navigation_radio")


# Home Page
if page == "Home":
    st.title("Health Insurance Premium Prediction")
    st.write("Welcome to the Health Insurance Premium Prediction app! This application helps users estimate their insurance premium based on key factors such as age, BMI, smoking status, and more.")
    st.image("healthinsimg.png", width=400)  # Replace with an actual image URL
    st.write("### How It Works:")
    st.write("1. Navigate to the **Predict** page and enter your details.")
    st.write("2. Click on **Predict Premium** to get an estimate.")
    st.write("3. Check your previous predictions in the **History** section.")
    st.write("4. Learn more about the project in the **About** page.")
# Predict Page
# elif page == "Predict":
#     st.title("Enter Your Details & Get Prediction")
#     age = st.number_input("Age", min_value=18, max_value=100, step=1, value=None, placeholder="Enter your age")
#     sex = st.selectbox("Gender", ["","Male", "Female"], index=0)
#     bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, step=0.1, value=None, placeholder="Enter your BMI")
#     children = st.number_input("Children", min_value=0, max_value=10, step=1, value=None, placeholder="Enter number of children")
#     smoker = st.selectbox("Smoker", ["", "Yes", "No"], index=0)
#     region = st.selectbox("Region", ["", "Southwest", "Southeast", "Northwest", "Northeast"], index=0)
    
#     if st.button("Predict Premium"):
#         if not age or not sex or not bmi or not smoker or not region or children<0:
#             st.error("All fields are required.")
#         else:
#             sex = 1 if sex == "Male" else 0
#             smoker = 1 if smoker == "Yes" else 0
#             region = encode_region(region)
#             prediction = predict_charges(age, sex, bmi, children, smoker, region)
#             st.write(f"### Estimated Premium: ${prediction:.2f}")
#             if "history" not in st.session_state:
#                 st.session_state.history = []
#             st.session_state.history.append({"Age":age, "Gender":sex, "BMI":bmi, "Children":children, "Smoker":smoker, "Region":region, "Prediction":prediction})

elif page == "Predict":
    st.title("Enter Your Details & Get Prediction")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, step=1, value=None, placeholder="Enter your age")
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, step=0.1, value=None, placeholder="Enter your BMI")
        children = st.number_input("Children", min_value=0, max_value=10, step=1, value=None, placeholder="Enter number of children")
    with col2:
        sex = st.selectbox("Gender", ["", "Male", "Female"], index=0)
        smoker = st.selectbox("Smoker", ["", "Yes", "No"], index=0)
        region = st.selectbox("Region", ["", "Southwest", "Southeast", "Northwest", "Northeast"], index=0)
    
    if st.button("Predict Premium"):
        if not age or not sex or not bmi or not smoker or not region or children<0:
            st.error("All fields are required.")
        else:
            sex = 1 if sex == "Male" else 0
            smoker = 1 if smoker == "Yes" else 0
            region = encode_region(region)
            prediction = predict_charges(age, sex, bmi, children, smoker, region)
            st.write(f"### Estimated Premium: ${prediction:.2f}")
            if "history" not in st.session_state:
                st.session_state.history = []
            st.session_state.history.append({"Age": age, "Gender": sex, "BMI": bmi, "Children": children, "Smoker": smoker, "Region": region, "Prediction": prediction})

# History Page
elif page == "History":
    st.title("Prediction History")
    if "history" in st.session_state and st.session_state.history:
        his=[]
        reg = {0:"southwest", 1:"southeast", 2:"northwest", 3:"northeast"}
        for i in st.session_state.history:
            Age = i["Age"]
            Gender = 'Male' if i["Gender"]==1 else 'Female'
            BMI = i["BMI"]
            Children = i["Children"]
            Smoker = 'Yes' if i["Smoker"]==1 else 'No'
            Region =reg[i["Region"]] 
            Premium = i["Prediction"]
            his.append((Age,Gender,BMI,Children,Smoker,Region,Premium))
        history_df = pd.DataFrame(his, columns=["Age", "Gender", "BMI", "Children", "Smoker", "Region", "Premium"])
        st.dataframe(history_df)
    else:
        st.write("No history available.")

    # Model Statistics Page
elif page == "Model Statistics":
    st.title("Model Statistics")
    st.write("This section provides details about the machine learning model used in this application.")
    
    # Load dataset
    df = pd.read_csv("Health_insurance.csv")
    
    st.write("### Dataset Overview:")
    st.write(f"- **Dataset Size:** {df.shape[0]} rows, {df.shape[1]} columns")
    st.write("- **Preview of Dataset:**")
    st.dataframe(df.head(5))
    
    st.write("### Model Details:")
    st.write("- **Algorithm Used:** Linear Regression")
    st.write("- **Accuracy:**  78%")
    st.write("- **Training Data Size:** 80% of total dataset")
    st.write("- **Testing Data Size:** 20% of total dataset")
    
    st.write("### Model Features:")
    st.write("The model is trained using the following features:")
    st.write("- **Age**: Age of the policyholder")
    st.write("- **Gender**: Gender of the policyholder (Encoded as 0 for Female, 1 for Male)")
    st.write("- **BMI**: Body Mass Index")
    st.write("- **Children**: Number of children covered by health insurance")
    st.write("- **Smoker**: Whether the policyholder is a smoker (0: No, 1: Yes)")
    st.write("- **Region**: Encoded as categorical variable")
    
    st.write("### Model Performance Metrics:")
    st.write("- **Mean Absolute Error (MAE):** 4186")
    st.write("- **Mean Squared Error (MSE):** 33635210")
   


# About Page
elif page == "About":
    st.title("About this Project")
    st.write("This project is created to predict health insurance premiums based on various input parameters like age, sex, BMI, number of children, smoking status, and region.")
    
    st.write("### Technologies Used:")
    st.write("- **Machine Learning Model:** Trained using Python with libraries like scikit-learn, pandas, and NumPy.")
    st.write("- **Frontend:** Developed using Streamlit for an interactive UI.")
            
    st.write("- **Deployment:** Can be hosted on platforms like Heroku or Streamlit Cloud.")
    
    st.write("### Libraries Used:")
    st.write("- **pandas**: For data manipulation and preprocessing.")
    st.write("- **NumPy**: Used for numerical computations.")
    st.write("- **scikit-learn**: Used for machine learning model training and evaluation.")
    st.write("- **matplotlib & seaborn**: For data visualization.")
    st.write("- **Streamlit**: To build the UI and create an interactive web application.")
    
    st.write("### Preprocessing Steps:")
    st.write("1. **Handling Missing Values:** The dataset is checked for missing values, and necessary imputation techniques are applied.")
    st.write("2. **Encoding Categorical Variables:** Features like 'sex', 'smoker', and 'region' are converted into numerical values.")
    st.write("3. **Feature Scaling:** Numerical features like BMI and age are normalized to ensure proper model training.")
    st.write("4. **Outlier Detection and Removal:** We analyze and remove extreme outliers to improve model accuracy.")
    st.write("5. **Data Splitting:** The dataset is split into training and testing sets to evaluate model performance.")
