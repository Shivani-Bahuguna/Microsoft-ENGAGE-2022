# -*- coding: utf-8 -*-
"""
Created on Tue May 24 23:49:15 2022

@author: vinay
"""
"""Data Cleaning Phase"""
import numpy as np
import pandas as pd

#data preprocessing
df=pd.read_csv("cars_engage_2022.csv")
df=df.drop(columns=['Unnamed: 0'])
#print(type(df['Make'].iloc[0]))

#choosing appropriate features
dfnew=df[['Make','Model','Variant','Ex-Showroom_Price','Fuel_Type','Fuel_Tank_Capacity','Body_Type','Gears','Seating_Capacity','Ventilation_System','Airbags','Navigation_System']].copy()

#transforming mileage attributes into one
dfnew['Model_Name']=[str(dfnew['Make'].iloc[i])+" "+str(dfnew['Model'].iloc[i]) for i in range(len(dfnew))]
mil=[]

for i in range(len(df)):
    #print(i)
    try:
        if pd.notnull(df['ARAI_Certified_Mileage'].iloc[i]):
            mil.append(float(str(df['ARAI_Certified_Mileage'].iloc[i].split()[0])))
        else:
            mil.append(float(str(df['ARAI_Certified_Mileage_for_CNG'].iloc[i].split()[0])))
    except Exception as e:
        dfnew=dfnew.drop(i)
        
dfnew['Mileage']=mil

#transforming/type casting/discretizing other attributes
dfnew['Airbags'].fillna("None",inplace=True)
dfnew['Ventilation_System'].fillna("None",inplace=True)
dfnew['Navigation_System'].fillna("None",inplace=True)
#drop rows with null values
dfnew=dfnew.dropna()

dfnew['Fuel_Capacity']=[float(dfnew['Fuel_Tank_Capacity'].iloc[i].split()[0]) for i in range(len(dfnew))]

price=[]
for i in range(len(dfnew)):
    dfnew['Seating_Capacity'].iloc[i]=str(int(dfnew['Seating_Capacity'].iloc[i]))
    if dfnew['Airbags'].iloc[i]!="None":
        dfnew['Airbags'].iloc[i]="Yes"
    tmp=dfnew['Ex-Showroom_Price'].iloc[i].split()[1]
    t=""
    for j in tmp.split(','):
        t=t+str(j)
    price.append(int(t))
dfnew['Price']=price

dfnew=dfnew.drop(['Make','Model','Variant','Ex-Showroom_Price','Fuel_Tank_Capacity'],axis=1)

#save transformed data
#dfnew.to_csv("cars_datafinal.csv")

"""Analysis and Prediction Phase"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

x=dfnew[['Fuel_Type' ,'Body_Type', 'Gears' ,'Seating_Capacity', 'Ventilation_System', 'Airbags', 'Navigation_System' , 'Mileage' ,'Fuel_Capacity', 'Price']]
y=dfnew['Model_Name']

le1=LabelEncoder()
le2=LabelEncoder()

for i in x.columns.values:
    if i=='Price' or i=='Mileage' or i=='Fuel_Capacity':
        continue
    else:
        #x[i]=x[i].astype('category')
        x[i]=le1.fit_transform(x[i])
#y=y.astype('category')
y=le2.fit_transform(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=1)
    
model=DecisionTreeClassifier()
model=model.fit(x_train,y_train)


"""Integration into website"""
import pickle

pickle.dump(model,open('model.pkl','wb'))   

from flask import Flask, request, render_template
import pickle

#Initialize the flask App
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

#default page of our web-app
@app.route('/')
def home():
    return render_template('webpage.html')
    
#To use the predict button in our web-app
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [float(x) for x in request.form.values()]
    features=np.array(features,dtype='float32')
 
    prediction = model.predict(features.reshape(1,10))
    
    ans= str(np.unique(dfnew['Model_Name'].values[int(prediction)]))
    
    return render_template('webpage.html',_car_type='Recommended Car for you is : {}'.format(ans))
    #return render_template('webpage.html', prediction_text='CO2 Emission of the vehicle is :{}'.format(final_features))

if __name__ == "__main__":
    app.run()




