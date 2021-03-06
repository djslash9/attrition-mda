#Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import plotly.express as px #for visualization

from PIL import Image

#load the model from disk
#import joblib
#model = joblib.load(r"./notebook/model1.sav")
import pickle

#Import python scripts
from preprocessing import preprocess

from scipy import stats
from scipy.stats import chi2_contingency, randint, norm
# from PIL import Image
from click import progressbar

from streamlit_option_menu import option_menu

#Load the model
with open('./notebook/model.pkl', 'rb') as fp:
    model = pickle.load(fp)

def main():

    def navigation():
        try:
            path = st.experimental_get_query_params()['p'][0]
        except Exception as e:
            st.error('Please use the main app.')
            return None
        return 
    
    def convert_df_to_csv(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')    
    
    #Defining bar chart function
    def bar(feature):
        #Groupby the categorical feature
        temp_df = df.groupby([feature, 'Attrition']).size().reset_index()
        temp_df = temp_df.rename(columns={0:'Count'})
        #Calculate the value counts of each distribution and it's corresponding Percentages
        value_counts_df = df[feature].value_counts().to_frame().reset_index()
        categories = [cat[1][0] for cat in value_counts_df.iterrows()]
        #Calculate the value counts of each distribution and it's corresponding Percentages
        num_list = [num[1][1] for num in value_counts_df.iterrows()]
        div_list = [element / sum(num_list) for element in num_list]
        percentage = [round(element * 100,1) for element in div_list]
        #Defining string formatting for graph annotation
        #Numeric section
        def num_format(list_instance):
            formatted_str = ''
            for index,num in enumerate(list_instance):
                if index < len(list_instance)-2:
                    formatted_str=formatted_str+f'{num}%, ' #append to empty string(formatted_str)
                elif index == len(list_instance)-2:
                    formatted_str=formatted_str+f'{num}% & '
                else:
                    formatted_str=formatted_str+f'{num}%'
            return formatted_str
        #Categorical section
        def str_format(list_instance):
            formatted_str = ''
            for index, cat in enumerate(list_instance):
                if index < len(list_instance)-2:
                    formatted_str=formatted_str+f'{cat}, '
                elif index == len(list_instance)-2:
                    formatted_str=formatted_str+f'{cat} & '
                else:
                    formatted_str=formatted_str+f'{cat}'
            return formatted_str

        #Running the formatting functions
        num_str = num_format(percentage)
        cat_str = str_format(categories)
        
        #Setting graph framework
        fig = px.bar(temp_df, x=feature, y='Count', color='Attrition', title=f'Attrition rate by {feature}', barmode="group", color_discrete_sequence=["green", "red"])
        fig.add_annotation(
                    text=f'Value count of distribution of {cat_str} are<br>{num_str} percentage respectively.',
                    align='left',
                    showarrow=False,
                    xref='paper',
                    yref='paper',
                    x=0,
                    y=1.14,
                    bordercolor='black',
                    borderwidth=1)
        fig.update_layout(
            # margin space for the annotations on the right
            margin=dict(r=100),
        )
        
        fig
 
    st.set_page_config(page_title='Employee Attrition Prediction App', layout = 'wide', initial_sidebar_state = 'auto')
# favicon being an object of the same kind as the one you should provide st.image() with (ie. a PIL array for example) or a string (url or local file path)
    col1, col2, col3 = st.columns(3)
    
    #Setting Application title
    st.title('Employee Attrition Prediction App')

    #Setting Application sidebar default
    # image = Image.open('App.jpg')
    #Setting Application title
    # st.sidebar.image(image)    
    # with st.sidebar:
    #    selected2 = option_menu("Main Menu", ['Explore the Database', 'Online', 'Batch'], 
    #        icons=['house', 'gear', 'cast'], menu_icon="circle", default_index=0)
    #    selected2
    # st.sidebar.info('This app is created to predict Employee Attition')

    #selected = option_menu("", ['Explore', 'Online', 'Batch'], 
    #    icons=['house', 'cloud-upload', "list-task"], 
    #    default_index=0, orientation="horizontal")
    #selected

    selected = option_menu("", ['Explore', 'Single', 'Batch'],
        icons=['house', 'cloud-upload', "list-task"], 
        menu_icon="cast", default_index=0, orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "orange", "font-size": "25px"}, 
            "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "green"},
        }
    )

    if selected == "Explore":
        # st.info("Explore the dataset below")
        #Based on our optimal features selection
        # st.subheader("Dataset Overview")
        # Dropdown menu to select a dataset
        st.sidebar.write("## Explore The Dataset")
        st.sidebar.write("### Select an option from below: ")
        
        #df = pd.read_csv(selected_dataset)
        df = pd.read_csv(r"./data/hr_data_cleaned.csv")
        selected_dataset = df
        # dataset_type = check_dataset_category(selected_dataset)
        # st.info(f'Dataset Type: {dataset_type}')

        # Show the dimension of the dataframe
        if st.sidebar.checkbox("Show number of rows and columns"):
            st.subheader('Number of rows and columns')
            st.warning(f'Rows: {df.shape[0]}')
            st.info(f'Columns: {df.shape[1]}')
        
            
        # Distribution of Attrition
        if st.sidebar.checkbox('Distribution of Attrition'):
            st.subheader('Distribution of Attrition')
            target_instance = df["Attrition"].value_counts().to_frame()
            target_instance = target_instance.reset_index()
            target_instance = target_instance.rename(columns={'index': 'Category'})
            fig = px.pie(target_instance, values='Attrition', names='Category', color_discrete_sequence=["green", "red"])
            fig
               
        # display the dataset
        if st.sidebar.checkbox("Show Dataset"):
            st.write("#### The Dataset")
            rows = st.number_input("Enter the number of rows to view", min_value=0,value=5)
            if rows > 0:
                st.dataframe(df.head(rows))  

        # Show dataset description
        if st.sidebar.checkbox("Show description of dataset"):
            st.write("#### The Description the Dataset")
            st.write(df.describe())
                
        # Select columns to display
        if st.sidebar.checkbox("Show dataset with selected columns"):
            # get the list of columns
            columns = df.columns.tolist()
            st.write("#### Explore the dataset with column selected:")
            selected_cols = st.multiselect("Select desired columns", columns)
            if len(selected_cols) > 0:
                selected_df = df[selected_cols]
                st.dataframe(selected_df)            

        # check the data set columns
        if st.sidebar.checkbox("Show dataset columns"): 
            st.write("#### Columns of the dataset:")
            st.write(df.columns)

        # counts how many of each class we have
        if st.sidebar.checkbox("Show dataset types"): 
            col_vals = df.select_dtypes(include=["object"])
            num_vals = df.select_dtypes(exclude=["object"])
            st.write("## Types of the dataset")
            
            st.warning("##### Numerical Values:")
            st.write(num_vals.columns)
            st.write("")
            st.warning("##### Categorical Values:")
            st.write(col_vals.columns)
            
   
        if st.sidebar.checkbox("Overlook the dataset"):
            st.write('### Overlook the dataset') 
            columns = df.columns.tolist()
            selected_cols = st.selectbox("Select the attribute:", columns)
            plt.title(selected_cols)
            fig = plt.figure(figsize=(10,6))
            plt.xticks(rotation=60, fontsize=10)   
            sns.countplot(x=selected_cols, data=df)
            st.pyplot(fig)  

        # Compare with Attrition    
        if st.sidebar.checkbox('Compare with Attrition'):
            st.write('#### Compare the dataset Against the Attrition')
            df_drop = df.drop('Attrition', axis=1)
            columns = df_drop.columns.tolist()
            selected_cols = st.selectbox("Select the attribute", columns)
            plt.title(selected_cols)             
            bar(selected_cols) 
            with st.expander("See sugessions"):
                if selected_cols == "Age":
                    st.write("""Employees are between *26 and 35* are more tend to be leaving the company soon. 
                             So performe survey to get their feedback.""")
                elif selected_cols == "BusinessTravel":
                    st.write("""Rarely Traveled Employees are more tend to be leaving the company soon. 
                             So give them an oppotunity to have a business travel soon.""")
                elif selected_cols == "Department":
                    st.write("""Employees of R&D are more tend to be leaving the company soon, followed by Sales dept. 
                             So guide department heads to have a discussion with employees.""")
                elif selected_cols == "DistanceFromHome":
                    st.write("""Employees who are near less than 3km from the office are tend to be leaving the company soon. 
                             So better having a face to face discussion with them to get to know about the job satisfaction.""")            
                elif selected_cols == "Education":
                    st.write("""Employees who hold bachelor or master degree are tend to be leaving the company soon. 
                             So performe a survey their job satisfaction factors and find solutions.""") 
                elif selected_cols == "EducationField":
                    st.write("""Employees who hold *Technical Degree, Life Science & Medical field* are tend to be leaving the company soon. 
                             So performe a survey for their job satisfaction factors and find solutions.""") 
                elif selected_cols == "EducationField":
                    st.write("""Employees who hold *Technical Degree, Life Science & Medical field* are tend to be leaving the company soon. 
                             So performe a survey for their job satisfaction factors and find solutions.""")                    
                elif selected_cols == "Gender":
                    st.write("""Most of *male employees* are tend to be leaving the company soon. 
                             So performe a survey for their job satisfaction factors and find solutions.""")     
                elif selected_cols == "JobInvolvement":
                    st.write("""Employees who have *high job involment rate* are tend to be leaving the company soon. 
                             So check their job involvement and share duties with other relevant employees.""")         
                elif selected_cols == "JobLevel":
                    st.write("""Employees who are in the *Junior Level* are tend to be leaving the company soon. 
                             So make necessary promotions on time.""")    
                elif selected_cols == "MaritalStatus":
                    st.write("""Employees who are *Single* are tend to be leaving the company soon. 
                             So performe a survey for their job satisfaction factors and find solutions.""")   
                elif selected_cols == "NumCompaniesWorked":
                    st.write("""Employees who worked for *less than 2 companies* before joining are tend to be leaving the company soon. 
                             So performe a survey for their job satisfaction factors and find solutions.""")
                elif selected_cols == "OverTime":
                    st.write("""Employees who are *having Over Time* are tend to be leaving the company soon. 
                             So share their duties and let them leave the office on time.""")    
                elif selected_cols == "PercentSalaryHike":
                    st.write("""Employees who are *having less than '15%' Salary Hike* are tend to be leaving the company soon. 
                             So make them satisfy by giving them an appropriate salary increment.""")        
                elif selected_cols == "PerformanceRating":
                    st.write("""Employees who are having *high Performance Rating* are tend to be leaving the company soon. 
                             So performe a survey for their job satisfaction factors and find solutions.""")     
                elif selected_cols == "StockOptionLevel":
                    st.write("""Employees who are having *No Stock Option Level* are tend to be leaving the company soon. 
                             So make their level of stock options high.""")       
                elif selected_cols == "TotalWorkingYears":
                    st.write("""Employees who are *less Than 10 Years* in the company are tend to be leaving the company soon. 
                             So performe a survey for their job satisfaction factors and find solutions.""")       
                elif selected_cols == "WorkLifeBalance":
                    st.write("""Employees who have *a Better Work Life Balance* are tend to be leaving the company soon. 
                             So performe a survey for their job satisfaction factors and find solutions.""")  
                elif selected_cols == "YearsAtCompany":
                    st.write("""Employees who are in the company *less Than 5 Years* are tend to be leaving the company soon. 
                             So performe a survey for their job satisfaction factors and find solutions.""")  
                elif selected_cols == "TrainingTimesLastYear":
                    st.write("""Employees who had *2 or 3 training times last year* are tend to be leaving the company soon. 
                             So performe a survey for their job satisfaction factors and find solutions.""") 
                elif selected_cols == "YearsSinceLastPromotion":
                    st.write("""Employees who are *not promoted last year* are tend to be leaving the company soon. 
                             So guide department heads to have a discussion with employees.""")
                else:
                    st.write("""Observe the graph and get necessary actions to reduce the attrition.""")   
                    
                                              
    elif selected == "Single":
        st.info("Input data below")
        #Based on our optimal features selection
        st.write("### Demographic data")
        st.write("#### Personal data")

        c1, c2, c3= st.columns(3)
        with c1:
            Gender = st.radio('Gender:', ('Male', 'Female'))
        if Gender == "Male":
            Gender = 1
        else:
            Gender = 2
        
        with c2:
            BusinessTravel = st.radio('Business Travel:', ('Travel_Rarely', 'Travel_Frequently', 'Non-Travel'))
        if BusinessTravel == "Travel_Rarely":
            BusinessTravel = 1
        elif BusinessTravel == 'Travel_Frequently':
            BusinessTravel = 2
        else:
            BusinessTravel = 3
            
        with c3:
            Department = st.radio('Department:', ('Sales', 'Research & Development', 'Human Resources'))            
        if Department == "Sales":
            Department = 1
        elif Department == 'Research & Development':
            Department = 2
        else:
            Department = 3
        
        
        c1, c2, c3= st.columns(3)    
        with c1:        
            EducationField = st.selectbox('Education Field', ('Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Other'))
        if EducationField == "Life Sciences":
            EducationField = 1
        elif EducationField == 'Medical':
            EducationField = 2
        elif EducationField == 'Marketing':
            EducationField = 3
        elif EducationField == 'Technical Degree':
            EducationField = 4   
        else:
            EducationField = 5         
        with c2:  
            EnvironmentSatisfaction = st.selectbox('Environment Satisfaction', ('Low', 'Medium', 'High', 'Very High'))
        if EnvironmentSatisfaction == "Low":
            EnvironmentSatisfaction = 1
        elif EnvironmentSatisfaction == 'Medium':
            EnvironmentSatisfaction = 2
        elif EnvironmentSatisfaction == 'High':
            EnvironmentSatisfaction = 3
        else:
            EnvironmentSatisfaction = 4
        with c3:             
            JobInvolvement = st.selectbox('Job Involvement', ('Low', 'Medium', 'High', 'Very High'))
        if JobInvolvement == "Low":
            JobInvolvement = 1
        elif JobInvolvement == 'Medium':
            JobInvolvement = 2
        elif JobInvolvement == 'High':
            JobInvolvement = 3
        else:
            JobInvolvement = 4
        c1, c2, c3= st.columns(3)              
        with c1:              
            JobLevel = st.selectbox('Job Level', ('Junior', 'Mid', 'Senior', 'Manager', 'Director'))
        if JobLevel == "Junior":
            JobLevel = 1
        elif JobLevel == 'Mid':
            JobLevel = 2
        elif JobLevel == 'Senior':
            JobLevel = 3
        elif JobLevel == 'Manager':
            JobLevel = 4            
        else:
            JobLevel = 5
                    
        # JobRole = st.selectbox('Job Role', ('Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'))
        with c2:  
            JobSatisfaction = st.selectbox('Job Satisfaction', ('Low', 'Medium', 'High', 'Very High'))
        if JobSatisfaction == "Low":
            JobSatisfaction = 1
        elif JobSatisfaction == 'Medium':
            JobSatisfaction = 2
        elif JobSatisfaction == 'High':
            JobSatisfaction = 3
        else:
            JobSatisfaction = 4    
            
        # MaritalStatus = st.radio('Marital Status:', ('Single', 'Married', 'Divorced'))
        with c1:  
            PerformanceRating = st.radio('Performance Rating:', ('Low', 'Good', 'Excellent', 'Outstanding'))
        if PerformanceRating == "Low":
            PerformanceRating = 1
        elif PerformanceRating == 'Good':
            PerformanceRating = 2
        elif PerformanceRating == 'Excellent':
            PerformanceRating = 3
        else:
            PerformanceRating = 4           
        with c3:  
            RelationshipSatisfaction = st.selectbox('Relationship Satisfaction', ('Low', 'Medium', 'High', 'Very High'))
        if RelationshipSatisfaction == "Low":
            RelationshipSatisfaction = 1
        elif RelationshipSatisfaction == 'Medium':
            RelationshipSatisfaction = 2
        elif RelationshipSatisfaction == 'High':
            RelationshipSatisfaction = 3
        else:
            RelationshipSatisfaction = 4 
        with c3:              
            WorkLifeBalance = st.selectbox('Work Life Balance', ('Bad', 'Good', 'Better', 'Best'))
        if WorkLifeBalance == "Bad":
            WorkLifeBalance = 1
        elif WorkLifeBalance == 'Good':
            WorkLifeBalance = 2
        elif WorkLifeBalance == 'Better':
            WorkLifeBalance = 3
        else:
            WorkLifeBalance = 4 
        with c2:             
            OverTime = st.radio('Over Time:', ('Yes', 'No'))
        
        st.sidebar.subheader("Numerical data") 
        Age = st.sidebar.slider('Age', 18, 65, 41)
        DistanceFromHome = st.sidebar.slider('Distance From Home', 0, 30, 10)
        MonthlyIncome = st.sidebar.slider('Monthly Income', 1000, 20000, 8000)
        NumCompaniesWorked = st.sidebar.slider('Num Companies Worked', 0, 10, 3)
        StockOptionLevel = st.sidebar.slider('Stock Option Level', 0, 3, 1)
        TotalWorkingYears = st.sidebar.slider('Total Working Years', 0, 40, 10)
        TrainingTimesLastYear = st.sidebar.slider('Training Times Last Year', 0, 6, 3)
        YearsAtCompany = st.sidebar.slider('Years At Company', 0, 40, 10)
        YearsInCurrentRole = st.sidebar.slider('Years In Current Role', 0, 18, 5)
        YearsSinceLastPromotion = st.sidebar.slider('Year Since Last Promotion', 0, 15, 4)
        YearsWithCurrManager = st.sidebar.slider('Years With Currunt Manager', 0, 20, 5)     
        DailyRate = st.sidebar.slider('Daily Rate', 100, 500, 1500) 
        
        data = {
            'BusinessTravel' : BusinessTravel,
            'Department' : Department,
            'EducationField' : EducationField,
            # 'MaritalStatus' : MaritalStatus,
            'Age' : Age,
            'DailyRate' : DailyRate,
            'DistanceFromHome' : DistanceFromHome,
            'EnvironmentSatisfaction' : EnvironmentSatisfaction, 
            'Gender' : Gender,
            # 'JobRole' : JobRole,
            'JobInvolvement' : JobInvolvement,
            'JobLevel' : JobLevel,
            'JobSatisfaction' : JobSatisfaction,
            'MonthlyIncome' : MonthlyIncome,
            'NumCompaniesWorked' : NumCompaniesWorked,
            'OverTime' : OverTime,
            'RelationshipSatisfaction' : RelationshipSatisfaction,
            'StockOptionLevel' : StockOptionLevel,
            'TotalWorkingYears' : TotalWorkingYears,
            'TrainingTimesLastYear' : TrainingTimesLastYear,
            'WorkLifeBalance' : WorkLifeBalance,
            'YearsAtCompany' : YearsAtCompany,
            'YearsInCurrentRole' : YearsInCurrentRole,
            'YearsSinceLastPromotion' : YearsSinceLastPromotion,
            'YearsWithCurrManager' : YearsWithCurrManager                      
        }

        features_df = pd.DataFrame.from_dict([data])
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.write('Overview of input is shown below')
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.dataframe(features_df)

        #Preprocess inputs
        preprocess_df = preprocess(features_df, 'Single')
        
    
        # st.write(prediction_df)
        
        if st.button('Predict'):
            prediction = model.predict(preprocess_df)
            prediction_df = pd.DataFrame(prediction, columns=["Predictions"])             
            
            if prediction == 1:
                st.warning('The Employee is going to leave the company soon, you may get necessary actions')
            elif prediction == 0:
                st.success('The Employee is not going to leave the company right now!')
        
    elif selected == "Batch":
        st.subheader("Dataset upload")
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            #Get overview of data
            st.write(data.head())
            st.markdown("<h3></h3>", unsafe_allow_html=True)
            #Preprocess inputs
            preprocess_df = preprocess(data, "Batch")
            if st.button('Predict'):
                #Get batch prediction
                prediction = model.predict(preprocess_df)
                prediction_df = pd.DataFrame(prediction, columns=["Predictions"])
                prediction_df = prediction_df.replace({1:'Yes', 0:'No'})

                st.markdown("<h3></h3>", unsafe_allow_html=True)
                st.subheader('Prediction')
                df_prep = pd.concat([prediction_df, data], axis=1)
                st.write(df_prep)
                
                
                st.download_button(
                    label="Download data as CSV",
                    data=convert_df_to_csv(df_prep),
                    file_name='predictions.csv',
                    mime='text/csv',
                    )                

    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

      #Setting Application description
    st.markdown("""
     :dart:  This Machine Learning App was implemented to predict employees leaving ability in an organization with provided dataset.
    The application is functional for both single prediction and batch data prediction. \n
    This is built for Master of Data Analytics final assesment of University of Kelaniya \n
    ###### by Sisira Dharmasena @djslash9
    """)
    st.markdown("<h3></h3>", unsafe_allow_html=True)            
if __name__ == '__main__':
    main()
