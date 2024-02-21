import pandas as pd

from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.metrics import confusion_matrix,accuracy_score,ConfusionMatrixDisplay
df=pd.read_csv('german_data.csv')
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

import pickle
pickle.dump(pipe,open('creditability.pk1','wb'))
loaded_model=pickle.load(open('creditability.pk1','rb'))

result=loaded_model.score(x_test,y_test)
print(result)
