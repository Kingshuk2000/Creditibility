import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df = pd.read_csv('german_data.csv')
df.replace({'Account_Balance' : {1:'No Account',2:'None',3:'Below 200 DM',4:'200 DM or Above'}},inplace=True)
df.replace({'Payment_Status_of_Previous_Credit': {0:'Delayed',1:'Other Credits',
                                                  2:'Paid up',3:'No Problem with Current Credits',
                                                  4:'Previous Credits Paid'}},inplace=True)
df.replace({'Value_Savings_Stocks' : {1 : 'None', 2 : 'Below 100 DM',
                                                          3: '[100, 500)',4:'[500, 1000)',5:'Above 1000'}},inplace = True)
df.replace({'Length_of_current_employment' : {1 : 'Unemployed', 2 : '<1 Year',
                                                          3: '[1, 4)',4:'[4, 7)',5:'Above 7'}},inplace=True)
df.replace({'Instalment_per_cent' : {1 : 'Above 35%', 2 : '(25%, 35%)',
                                                          3: '[20%, 25%)',4:'Below 20%'}},inplace=True)
df.replace({'Occupation' : {1 : 'Unemployed, unskilled', 2 : 'Unskilled Permanent Resident',
                                                          3: 'Skilled',4:'Executive'}},inplace=True)
df.replace({'Sex_Marital_Status' : {1 : 'Male, Divorced', 2 : 'Male, Single',
                                                          3: 'Male, Married/Widowed',4:'Female'}},inplace=True)
df.replace({'Duration_in_Current_address' : {1 : '<1 Year', 2 : '[1, 4)',
                                                          3: '[4, 7)',4:'Above 7'}},inplace=True)
df.replace({'Type_of_apartment' : {1 : 'Free', 2 : 'Rented',
                                                          3: 'Owned'}},inplace=True)
df.replace({'Most_valuable_available_asset' : {1 : 'None', 2 : 'Car',
                                                          3: 'Life Insurance',4:'Real Estate'}},inplace=True)
df.replace({'No_of_Credits_at_this_Bank' : {1 : '1', 2 : '2 or 3',
                                                          3: '4 or 5',4:'Above 6'}},inplace=True)
df.replace({'Guarantors' : {1 : 'None', 2 : 'Co-applicant',
                                                          3: 'Guarantor'}},inplace=True)
df.replace({'Concurrent_Credits' : {1 : 'Other Banks', 2 : 'Dept. Store',
                                                          3: 'None'}},inplace=True)
df.replace({'No_of_dependents' : {1 : '3 or More', 2 : 'Less than 3'}},inplace=True)
df.replace({'Telephone' : {1 : 'No', 2 : 'Yes'}},inplace=True)
df.replace({'Foreign_Worker' : {1 : 'No', 2 : 'Yes'}},inplace=True)
df.replace({'Purpose' : {0:'Other',1 : 'New Car', 2 : 'Used Car',
                               3:'Furniture',4:'Radio/TV',5:'Appliances',
                               6:'Repair',8:'Vacation',9:'Retraining',10:'Business'}},inplace=True)

X=df.iloc[:,1:]
Y=df.iloc[:,0]
trf1=ColumnTransformer([
    ('oe',OrdinalEncoder(),[0,2,3,5,6,7,8,9,10,11,13,14,15,16,17,18,19]),
],remainder='passthrough')
model=RandomForestClassifier()

x_train,x_test,y_train,y_test= train_test_split(X,Y,test_size=0.3,random_state=0,stratify=df['Creditability'])
pipe=Pipeline([
    ('trf1',trf1),
    ('model',model),
])
pipe.fit(x_train,y_train)









st.title('Creditability Project')



Account_Balance = st.selectbox('Account_Balance',('No Account','None','Below 200 DM','200 DM or Above'))
Duration_of_Credit_monthly=st.number_input('Duration_of_Credit_monthly',min_value=4,max_value=72)
st.write('range - 4 to 72  ')
Payment_Status_of_Previous_Credit = st.selectbox('Payment_Status_of_Previous_Credit',
                                                 ('Delayed','Other Credits',
                                                  'Paid up','No Problem with Current Credits',
                                                  'Previous Credits Paid'))
Value_Savings_Stocks = st.selectbox('Value_Savings_Stocks',
                                    ('None',  'Below 100 DM',
                                    '[100, 500)','[500, 1000)','Above 1000'))
Credit_Amount =st.number_input("Credit_Amount ",min_value=250,max_value=18424)
st.write('range - 200 to 19000  ')
Length_of_current_employment = st.selectbox('Length_of_current_employment',
                                            ( 'Unemployed', '<1 Year',
                                            '[1, 4)','[4, 7)','Above 7'))
Instalment_per_cent = st.selectbox('Instalment_per_cent',
                                   ('Above 35%', '(25%, 35%)',
                                    '[20%, 25%)','Below 20%'))
Occupation = st.selectbox('Occupation',('Unemployed, unskilled', 'Unskilled Permanent Resident',
                                        'Skilled','Executive'))
Sex_Marital_Status = st.selectbox('Sex_Marital_Status',
                                  ('Male, Divorced', 'Male, Single',
                                   'Male, Married/Widowed','Female'))
Duration_in_Current_address = st.selectbox('Duration_in_Current_address',
                                           ('<1 Year', '[1, 4)',
                                            '[4, 7)','Above 7'))
Type_of_apartment = st.selectbox('Type_of_apartment',
                                 ('Free','Rented','Owned'))
Most_valuable_available_asset = st.selectbox('Most_valuable_available_asset',
                                 ('None', 'Car','Life Insurance','Real Estate'))
Age_years=st.number_input("Age_years",min_value=19,max_value=75)
st.write('range - 19 to 75  ')
No_of_Credits_at_this_Bank = st.selectbox('No_of_Credits_at_this_Bank',
                                 ( '1','2 or 3','4 or 5','Above 6'))
Guarantors = st.selectbox('Guarantors',
                                 ('None','Co-applicant','Guarantor'))
Concurrent_Credits = st.selectbox('Concurrent_Credits',
                                 ('Other Banks','Dept. Store','None'))
No_of_dependents = st.selectbox('No_of_dependents',
                                 ( '3 or More', 'Less than 3'))
Telephone = st.selectbox('Telephone',
                                 ('No', 'Yes'))
Foreign_Worker = st.selectbox('Foreign_Worker',
                                 ('No','Yes'))
Purpose = st.selectbox('Purpose',
                                 ('Other','New Car', 'Used Car',
                               'Furniture','Radio/TV','Appliances',
                               'Repair','Vacation','Retraining','Business'))
user_input=pd.DataFrame(data=np.array([Account_Balance,Duration_of_Credit_monthly,
             Payment_Status_of_Previous_Credit,Purpose,Credit_Amount,Value_Savings_Stocks,
             Length_of_current_employment,Instalment_per_cent,Sex_Marital_Status,
             Guarantors,Duration_in_Current_address,Most_valuable_available_asset,
             Age_years,	Concurrent_Credits,	Type_of_apartment,	No_of_Credits_at_this_Bank	,
             Occupation,	No_of_dependents,	Telephone,	Foreign_Worker]).reshape(1,20),columns=X.columns)
st.write(user_input)
creditability=pipe.predict(user_input)

if st.button("Predict creditability"):
    creditability=pipe.predict(user_input)
    if creditability == 1:
        st.write('you are eligible for loans')
    else:
        st.write('you are not eligible for loans')
