#Import libraries
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

#load the model from disk
import joblib
model = joblib.load(r"./notebook/model.sav")

#Import python scripts
from preprocessing import preprocess

def main():
    def convert_df(df):
       return df.to_csv().encode('utf-8')
   
    #Setting Application title
    st.title('Employee Attrition Prediction App')

      #Setting Application description
    st.markdown("""
     :dart:  This app is made to predict about the people who are tend to leave the office for some reasons.
    The application is functional for both single prediction and batch data prediction. \n
    """)
    st.markdown("<h3></h3>", unsafe_allow_html=True)

    #Setting Application sidebar default
    image = Image.open('App.jpg')
    add_selectbox = st.sidebar.selectbox(
	"How would you like to predict?", ("Online", "Batch"))
    st.sidebar.info('This app is created to predict Employee Attrition')
    st.sidebar.image(image)

    
    if add_selectbox == "Online":
        st.subheader("Personal data")
        businessTravel = st.radio('Business Travel:', ('Travel_Rarely', 'Travel_Frequently', 'Non_Travel'))
        educationField = st.selectbox('Education Field', ('Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Other'))
        environmentSatisfaction = st.selectbox('Environment Satisfaction', ('Low', 'Medium', 'High', 'Very_High'))
        jobInvolvement = st.selectbox('Job Involvement', ('Low', 'Medium', 'High', 'Very_High'))
        jobLevel = st.selectbox('Job Level', ('Junior', 'Mid', 'Senior', 'Manager', 'Director'))
        jobRole = st.selectbox('Job Role', ('Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'))
        jobSatisfaction = st.selectbox('Job Satisfaction', ('Low', 'Medium', 'High', 'Very_High'))
        maritalStatus = st.radio('Marital Status:', ('Single', 'Married', 'Divorced'))
        overTime = st.radio('Over Time:', ('Yes', 'No'))
        relationshipSatisfaction = st.selectbox('Relationship Satisfaction', ('Low', 'Medium', 'High', 'Very_High'))
        workLifeBalance = st.selectbox('Work Life Balance', ('Bad', 'Good', 'Better', 'Best'))
        
        st.subheader("Professional data") 
        age = st.slider('Age', 0, 65, 25)
        distanceFromHome = st.slider('Distance From Home', 0, 30, 10)
        monthlyIncome = st.slider('Monthly Income', 1000, 20000, 8000)
        numCompaniesWorked = st.slider('Num Companies Worked', 0, 10, 3)
        stockOptionLevel = st.slider('Stock Option Level', 0, 3, 1)
        totalWorkingYears = st.slider('Total Working Years', 0, 40, 10)
        trainingTimesLastYear = st.slider('Training Times Last Year', 0, 6, 3)
        yearsAtCompany = st.slider('Years At Company', 0, 40, 10)
        yearsInCurrentRole = st.slider('Years In Current Role', 0, 18, 5)
        yearsSinceLastPromotion = st.slider('Year Since Last Promotion', 0, 15, 4)
        yearsWithCurrManager = st.slider('Years With Currunt Manager', 0, 20, 5) 
                
        data = {
            'BusinessTravel' : businessTravel,
            'EducationField' : educationField,
            'EnvironmentSatisfaction' : environmentSatisfaction,
            'JobInvolvement' : jobInvolvement,
            'JobLevel' : jobLevel,
            'JobRole' : jobRole,
            'JobSatisfaction' : jobSatisfaction,
            'MaritalStatus' : maritalStatus,
            'OverTime' : overTime,
            'RelationshipSatisfaction' : relationshipSatisfaction,
            'WorkLifeBalance' : workLifeBalance,
            'Age' : age,
            'DistanceFromHome' : distanceFromHome,
            'MonthlyIncome' : monthlyIncome,
            'NumCompaniesWorked' : numCompaniesWorked,
            'StockOptionLevel' : stockOptionLevel,
            'TotalWorkingYears' : totalWorkingYears,
            'TrainingTimesLastYear' : trainingTimesLastYear,
            'YearsAtCompany' : yearsAtCompany,
            'YearsInCurrentRole' : yearsInCurrentRole,
            'YearsSinceLastPromotion' : yearsSinceLastPromotion,
            'YearsWithCurrManager' : yearsWithCurrManager   
        }
        
        features_df = pd.DataFrame.from_dict([data])
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.write('Overview of input is shown below')
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.dataframe(features_df)


        preprocess_df = preprocess(features_df, 'Online')
        prediction = model.predict(preprocess_df)
       

        if st.button('Predict'):
         
            if prediction == 1:
                st.warning('Yes, the person will be leaving.')
            else:
                st.success('No, the person is happy.')
        

    else:
        st.subheader("Dataset upload")
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            
            #Get overview of data
            st.write(data.head())
            st.markdown("<h3></h3>", unsafe_allow_html=True)
            #Preprocess inputs
            data.dropna(inplace=True)
            if st.button('Predict'):
                #Get batch prediction
                preprocess_df = preprocess(data, "Batch")
                prediction = model.predict(preprocess_df)
                prediction_df = pd.DataFrame(prediction, columns=["Predictions"])
                prediction_df = prediction_df.replace({1:'Yes', 0:'No'})

                st.markdown("<h3></h3>", unsafe_allow_html=True)
                st.subheader('Prediction')
                st.write(prediction_df)
                
                st.markdown("<h3></h3>", unsafe_allow_html=True)
                csv = convert_df(prediction_df)
                st.download_button("Press to Download", csv, "hr_predicted.csv", "text/csv", key='download-csv')

    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
            
if __name__ == '__main__':
        main()