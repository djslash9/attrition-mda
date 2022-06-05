import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess(df, option):
    """
    This function is to cover all the preprocessing steps on the churn dataframe. It involves selecting 
    important features, encoding categorical data, handling missing values,feature scaling and splitting the data
    """
    # df['Gender'] = df['Gender'].map({'Male':1, 'Female':0})

    #Drop values based on operational options
    if (option == "Single"):
        columns = ['Age', 'DistanceFromHome', 'MonthlyIncome', 'NumCompaniesWorked', 'StockOptionLevel', 
                   'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole', 
                   'YearsSinceLastPromotion', 'YearsWithCurrManager', 'BusinessTravel_Travel_Frequently', 
                   'BusinessTravel_Travel_Rarely', 'EducationField_Life Sciences', 'EducationField_Medical', 
                   'EducationField_Other', 'EnvironmentSatisfaction_Low', 'JobInvolvement_Low', 'JobInvolvement_Very_High', 
                   'JobLevel_Junior', 'JobLevel_Manager', 'JobLevel_Mid', 'JobRole_Research Director', 
                   'JobRole_Research Scientist', 'JobRole_Sales Executive', 'JobSatisfaction_Very_High', 
                   'MaritalStatus_Single', 'OverTime_Yes', 'RelationshipSatisfaction_Low', 'WorkLifeBalance_Best', 
                   'WorkLifeBalance_Better', 'WorkLifeBalance_Good']

        #Encoding the other categorical categoric features with more than two categories
        # df = pd.get_dummies(df).reindex(columns=columns, fill_value=0)
        #df = pd.get_dummies(df, drop_first=True)
        df = pd.get_dummies(df, drop_first=True).reindex(columns=columns, fill_value=0)
     
    elif (option == "Batch"):
        pass
        df = df[['Age', 'DistanceFromHome', 'MonthlyIncome', 'NumCompaniesWorked', 'StockOptionLevel', 
                   'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole', 
                   'YearsSinceLastPromotion', 'YearsWithCurrManager', 'BusinessTravel', 'EducationField', 
                   'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction', 
                   'MaritalStatus', 'OverTime', 'RelationshipSatisfaction', 'WorkLifeBalance']]
        
        
        
        
        
        columns = ['Age', 'DistanceFromHome', 'MonthlyIncome', 'NumCompaniesWorked', 'StockOptionLevel', 
                   'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole', 
                   'YearsSinceLastPromotion', 'YearsWithCurrManager', 'BusinessTravel_Travel_Frequently', 
                   'BusinessTravel_Travel_Rarely', 'EducationField_Life Sciences', 'EducationField_Medical', 
                   'EducationField_Other', 'EnvironmentSatisfaction_Low', 'JobInvolvement_Low', 'JobInvolvement_Very_High', 
                   'JobLevel_Junior', 'JobLevel_Manager', 'JobLevel_Mid', 'JobRole_Research Director', 
                   'JobRole_Research Scientist', 'JobRole_Sales Executive', 'JobSatisfaction_Very_High', 
                   'MaritalStatus_Single', 'OverTime_Yes', 'RelationshipSatisfaction_Low', 'WorkLifeBalance_Best', 
                   'WorkLifeBalance_Better', 'WorkLifeBalance_Good']
        
        #Encoding the other categorical categoric features with more than two categories
        # df = pd.get_dummies(df, drop_first=True)
        df = pd.get_dummies(df, drop_first=True).reindex(columns=columns, fill_value=0)
    else:
        print("Incorrect operational options")

    #feature scaling
    sc = MinMaxScaler()
    scaled = sc.fit_transform(df)

    # gets the Data Frame version of numerical scaled for later manipulation
    df_scaled = pd.DataFrame(scaled)

    # renaming the columns of result Data Frame
    df_scaled.columns = df.columns
    df = df_scaled
    
    return df
