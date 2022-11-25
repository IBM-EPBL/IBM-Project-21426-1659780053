# main.py
import os
import base64
import io
import math
from flask import Flask, render_template, Response, redirect, request, session, abort, url_for
import mysql.connector
import hashlib
import datetime
import calendar
import random
from random import randint
from urllib.request import urlopen
import webbrowser

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image

import urllib.request
import urllib.parse
import socket    
import csv

import matplotlib.dates as mdates
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')


mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  charset="utf8",
  database="wind_turbine"

)
app = Flask(__name__)
##session key
app.secret_key = 'abcdef'
#######
UPLOAD_FOLDER = 'static/upload'
ALLOWED_EXTENSIONS = { 'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#####
@app.route('/', methods=['GET', 'POST'])
def index():
    msg=""

   
    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM admin WHERE username = %s AND password = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            return redirect(url_for('admin'))
        else:
            msg = 'Incorrect username/password!'
    return render_template('index.html',msg=msg)

@app.route('/login', methods=['GET', 'POST'])
def login():
    msg=""

   
    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM admin WHERE username = %s AND password = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            return redirect(url_for('admin'))
        else:
            msg = 'Incorrect username/password!'
    return render_template('login.html',msg=msg)

@app.route('/login_user', methods=['GET', 'POST'])
def login_user():
    msg=""

   
    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM wt_register WHERE uname = %s AND pass = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            return redirect(url_for('test_data'))
        else:
            msg = 'Incorrect username/password!'
    return render_template('login_user.html',msg=msg)

@app.route('/register', methods=['GET', 'POST'])
def register():
    msg=""
    act=request.args.get("act")
    if request.method=='POST':
        name=request.form['name']
        location=request.form['location']
        mobile=request.form['mobile']
        email=request.form['email']
        uname=request.form['uname']
        pass1=request.form['pass']
        
    
        
        mycursor = mydb.cursor()

        now = datetime.datetime.now()
        rdate=now.strftime("%d-%m-%Y")
    
        mycursor.execute("SELECT count(*) from wt_register where uname=%s",(uname,))
        cnt = mycursor.fetchone()[0]
    
        if cnt==0:
            mycursor.execute("SELECT max(id)+1 FROM wt_register")
            maxid = mycursor.fetchone()[0]
            if maxid is None:
                maxid=1
                    
            sql = "INSERT INTO wt_register(id,name,location,mobile,email,uname,pass,create_date) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
            val = (maxid,name,location,mobile,email,uname,pass1,rdate)
            mycursor.execute(sql, val)
            mydb.commit()            
            #print(mycursor.rowcount, "Registered Success")
            msg="sucess"
            #if mycursor.rowcount==1:
            return redirect(url_for('register',act='1'))
        else:
            msg='User Already Exist!'
    return render_template('register.html',msg=msg,act=act)

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    msg=""

   
    pd.pandas.set_option('display.max_columns', None)
    dataset = pd.read_csv('static/dataset/train.csv')
    dat=dataset.head()
    data=[]
    for ss in dat.values:
        data.append(ss)

        
    return render_template('admin.html',msg=msg,data=data)

@app.route('/load_data', methods=['GET', 'POST'])
def load_data():
    msg=""
    
    pd.pandas.set_option('display.max_columns', None)
    dataset = pd.read_csv('static/dataset/train.csv')
    dat=dataset.head(200)

    data=[]
    for ss in dat.values:
        data.append(ss)
    

    return render_template('load_data.html',data=data)


@app.route('/preprocess', methods=['GET', 'POST'])
def preprocess():
    msg=""

    pd.pandas.set_option('display.max_columns', None)
    dataset = pd.read_csv('static/dataset/train.csv')
    dat=dataset.head(200)
    rows=len(dataset.values)
    data3=[]
    for ss3 in dat.values:
        cnt=len(ss3)
       
        data3.append(ss3)
    cols=cnt
    mem=float(rows)*0.75

    ##
    list_of_column_names=[]
    with open("static/dataset/train.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        list_of_column_names = []
        for row in csv_reader:
            list_of_column_names.append(row)
            break
    ##

    print(list_of_column_names)
 
    dat4=dataset.isna().sum()
    dr=np.stack(dat4)
    #print(dr)
    
    data4=[]
    i=0
    k=len(dr)
    j=k-1
    for ss4 in dr:
        if i<j: 
            dt=[]
            dt.append(list_of_column_names[0][i])
            dt.append(ss4)
            data4.append(dt)
        i+=1 

    return render_template('preprocess.html',mem=mem,rows=rows,cols=cols,data3=data3,data4=data4)



@app.route('/data_analysis', methods=['GET', 'POST'])
def data_analysis():
    msg=""

    pd.pandas.set_option('display.max_columns', None)
    dataset = pd.read_csv('static/dataset/train.csv')
    dat=dataset.head(200)

    null_col = [i for i in dataset.columns if dataset[i].isnull().sum()>0 and dataset[i].dtypes != 'O']
    null_col

    for i in null_col:
        dataset[i].fillna(dataset[i].mean(), inplace=True)
    dataset.isnull().sum()

    dataset.cloud_level.value_counts()
    dataset['cloud_level'].fillna('Low', inplace=True)
    dataset.isnull().sum()

    dataset['turbine_status'].unique()
    dataset['turbine_status'].fillna('Missing', inplace=True)
    l = ['BA', 'A2', 'ABC', 'AAA', 'BD', 'AC', 'BB', 'BCB', 'B', 'AB', 'Missing', 'B2', 'BBB', 'A', 'D']
    feat_tur = dict()
    for i in range(len(l)):
        feat_tur[l[i]] = i
    feat_tur
    dataset['turbine_status']=dataset['turbine_status'].map(feat_tur)

    dataset.isnull().sum()
    dataset['datetime'] = pd.to_datetime(dataset['datetime'])
    dataset['day'] = dataset['datetime'].dt.date
    dataset['time'] = dataset['datetime'].dt.time
    # dataset.drop('datetime', axis=1, inplace=True)
    dataset['day'] = pd.to_datetime(dataset['day'])
    dataset['time']= pd.to_datetime(dataset['time'].astype(str))
    dataset.dtypes
    dataset['date']=dataset['day'].dt.day
    dataset['month']=dataset['day'].dt.month
    dataset['year']=dataset['day'].dt.year
    dataset.drop('day', axis=1, inplace=True)
    dataset.head(2)

    dataset['time_hour'] = dataset['time'].dt.hour
    dataset['time_minute'] = dataset['time'].dt.minute
    dataset.drop('time', axis=1, inplace=True)
    dataset.head(2)
    dataset.dtypes

    #
    plt_feat = ['datetime', 'wind_speed(m/s)',
       'atmospheric_temperature(°C)', 'shaft_temperature(°C)',
       'blades_angle(°)', 'gearbox_temperature(°C)', 'engine_temperature(°C)',
       'motor_torque(N-m)', 'generator_temperature(°C)',
       'atmospheric_pressure(Pascal)', 'area_temperature(°C)',
       'windmill_body_temperature(°C)', 'wind_direction(°)', 'resistance(ohm)',
       'rotor_torque(N-m)', 'cloud_level', 'blade_length(m)',
       'blade_breadth(m)', 'windmill_height(m)']

    '''import matplotlib.pyplot as plt
    import seaborn as sns
    for i in plt_feat:
        plt.figure(figsize=(8,4))
        sns.scatterplot(data=dataset, x=i, y='windmill_generated_power(kW/h)')
        plt.show()'''


    return render_template('data_analysis.html')



@app.route('/feature_extract', methods=['GET', 'POST'])
def feature_extract():
    msg=""

    df_train = pd.read_csv(r"static/dataset/train.csv")
    df_test= pd.read_csv(r"static/dataset/test.csv")
    df_train.head()
    df_train.nunique()
    df_test.nunique()
    df_train.isna().sum()
    df_test.isna().sum()
    dat=df_train.corr()
    #print(dat)
    col=['wind_speed(m/s)','atmospheric_temperature(C)','shaft_temperature(C)','blades_angle','gearbox_temperature(C)','engine_temperature(C)','motor_torque(N-m)','generator_temperature(C)','atmospheric_pressure(Pascal)','area_temperature(C)','windmill_body_temperature(C)','wind_direction','resistance(ohm)','rotor_torque(N-m)','blade_length(m)','blade_breadth(m)','windmill_height(m)']
    data=[]
    #col=list(df_train.columns)
    i=0
    for ss3 in dat.values:
        dt=[]
        if i<17:
            dt.append(col[i])
            dt.append(ss3)
            data.append(dt)
        i+=1
    
    #
    corr = df_train.corr()
    corr.style.background_gradient(cmap='coolwarm')
    def splitFeatures(df):
        numerical_features = df.select_dtypes(include=[np.number])
        categorical_features = df.select_dtypes(include=[np.object])
        return numerical_features, categorical_features

    numerical_features,categorical_features=splitFeatures(df_train)
    numerical_features
    dat2=categorical_features
    j=0
    data2=[]
    for ss2 in categorical_features.values:
        if i<200:
            data2.append(ss2)
        j+=1

    ####
    df_cpy = df_train.copy()
    def comparing_train_and_test_feature(df,df_test,col):
        fig = plt.figure(figsize=(16,10))
        ax0 = fig.add_subplot(1,2,1)
        ax1 = fig.add_subplot(1,2,2)
        df[col].plot(kind='kde',ax=ax0)
        df_test[col].plot(kind='kde',ax=ax1)
        ax0.set_xlabel(col)
        ax1.set_xlabel(col)
        ax0.set_title("Density plot of " + str(col) + " of training set")
        ax1.set_title("Density plot of " + str(col) + " of testing set")
        #plt.show()

    #comparing_train_and_test_feature(df_train,df_test,'wind_speed(m/s)')
    #sns.scatterplot(x='wind_speed(m/s)',y='windmill_generated_power(kW/h)',hue='cloud_level',data=df_train)
    #plt.show()

    #comparing_train_and_test_feature(df_train,df_test,'atmospheric_temperature(°C)')
    #sns.scatterplot(x='atmospheric_temperature(°C)',y='windmill_generated_power(kW/h)',hue='cloud_level',data=df_train)
    #plt.show()
    #plt.close()
    #comparing_train_and_test_feature(df_train,df_test,'shaft_temperature(°C)')
    '''sns.scatterplot(x='shaft_temperature(°C)',y='windmill_generated_power(kW/h)',hue='cloud_level',data=df_train)
    plt.show()
    plt.close()
    #comparing_train_and_test_feature(df_train,df_test,'blades_angle(°)')
    sns.scatterplot(x='gearbox_temperature(°C)',y='windmill_generated_power(kW/h)',hue='cloud_level',data=df_train)
    plt.show()
    plt.close()'''



    return render_template('feature_extract.html',data=data,data2=data2)


@app.route('/classify', methods=['GET', 'POST'])
def classify():
    msg=""

    train = pd.read_csv('static/dataset/train.csv')
    test = pd.read_csv('static/dataset/test.csv')
    train.head()
    test.head()
    train.describe()

    test.describe()
    ##
    corr = train.corr()
    plt.figure(figsize=(20,10))
    mask = np.zeros_like(corr,dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    #sns.heatmap(corr,mask=mask,annot=True)
    #plt.show()

    ###
    train.drop(['generator_temperature(°C)','windmill_body_temperature(°C)'],inplace=True,axis=1)
    test.drop(['generator_temperature(°C)','windmill_body_temperature(°C)'],inplace=True,axis=1)

    train.isnull().sum()
    test.isnull().sum()
    #sns.heatmap(train.isnull(),cbar=False,yticklabels=False,cmap = 'viridis')
    #sns.heatmap(test.isnull(),cbar=False,yticklabels=False,cmap = 'viridis')



    train['gearbox_temperature(°C)'].fillna(train['gearbox_temperature(°C)'].mean(),inplace=True)
    train['area_temperature(°C)'].fillna(train['area_temperature(°C)'].mean(),inplace=True)
    train['rotor_torque(N-m)'].fillna(train['rotor_torque(N-m)'].mean(),inplace=True)
    train['blade_length(m)'].fillna(train['blade_length(m)'].mean(),inplace=True)
    train['blade_breadth(m)'].fillna(train['blade_breadth(m)'].mean(),inplace=True)
    train['windmill_height(m)'].fillna(train['windmill_height(m)'].mean(),inplace=True)
    train['cloud_level'].fillna(train['cloud_level'].mode()[0],inplace=True)
    train['atmospheric_temperature(°C)'].fillna(train['atmospheric_temperature(°C)'].mean(),inplace=True)
    train['atmospheric_pressure(Pascal)'].fillna(train['atmospheric_pressure(Pascal)'].mean(),inplace=True)
    train['wind_speed(m/s)'].fillna(train['wind_speed(m/s)'].mean(),inplace=True)
    train['shaft_temperature(°C)'].fillna(train['shaft_temperature(°C)'].mean(),inplace=True)
    train['blades_angle(°)'].fillna(train['blades_angle(°)'].mean(),inplace=True)
    train['engine_temperature(°C)'].fillna(train['engine_temperature(°C)'].mean(),inplace=True)
    train['motor_torque(N-m)'].fillna(train['motor_torque(N-m)'].mean(),inplace=True)
    train['wind_direction(°)'].fillna(train['wind_direction(°)'].mean(),inplace=True)

    test['gearbox_temperature(°C)'].fillna(test['gearbox_temperature(°C)'].mean(),inplace=True)
    test['area_temperature(°C)'].fillna(test['area_temperature(°C)'].mean(),inplace=True)
    test['rotor_torque(N-m)'].fillna(test['rotor_torque(N-m)'].mean(),inplace=True)
    test['blade_length(m)'].fillna(test['blade_length(m)'].mean(),inplace=True)
    test['blade_breadth(m)'].fillna(test['blade_breadth(m)'].mean(),inplace=True)
    test['windmill_height(m)'].fillna(test['windmill_height(m)'].mean(),inplace=True)
    test['cloud_level'].fillna(test['cloud_level'].mode()[0],inplace=True)
    test['atmospheric_temperature(°C)'].fillna(test['atmospheric_temperature(°C)'].mean(),inplace=True)
    test['atmospheric_pressure(Pascal)'].fillna(test['atmospheric_pressure(Pascal)'].mean(),inplace=True)
    test['wind_speed(m/s)'].fillna(test['wind_speed(m/s)'].mean(),inplace=True)
    test['shaft_temperature(°C)'].fillna(test['shaft_temperature(°C)'].mean(),inplace=True)
    test['blades_angle(°)'].fillna(test['blades_angle(°)'].mean(),inplace=True)
    test['engine_temperature(°C)'].fillna(test['engine_temperature(°C)'].mean(),inplace=True)
    test['motor_torque(N-m)'].fillna(test['motor_torque(N-m)'].mean(),inplace=True)
    test['wind_direction(°)'].fillna(test['wind_direction(°)'].mean(),inplace=True)

    train.dropna(how='any',axis=0,inplace=True)

    train['cloud_level'].replace(['Extremely Low', 'Low', 'Medium'],[0, 1, 2],inplace=True)
    test['cloud_level'].replace(['Extremely Low', 'Low', 'Medium'],[0, 1, 2],inplace=True)
    train['turbine_status'].value_counts()
    test['turbine_status'].value_counts()
    dummy = ['turbine_status']
    train_dummy = pd.get_dummies(train[dummy])
    test_dummy = pd.get_dummies(test[dummy])
    train_dummy
    test_dummy
    train = pd.concat([train,train_dummy],axis=1)
    test = pd.concat([test,test_dummy],axis=1)

    train["datetime"] = pd.to_datetime(train["datetime"])
    test["datetime"] = pd.to_datetime(test["datetime"])

    train['dmonth'] = train['datetime'].dt.month
    train['dday'] = train['datetime'].dt.day
    train['ddayofweek'] = train['datetime'].dt.dayofweek

    test['dmonth'] = test['datetime'].dt.month
    test['dday'] = test['datetime'].dt.day
    test['ddayofweek'] = test['datetime'].dt.dayofweek

    ##Auto Regression
    '''X = train.drop(['tracking_id','datetime','windmill_generated_power(kW/h)','turbine_status'],axis=1)
    y = train['windmill_generated_power(kW/h)']

    print(X.shape, y.shape)

    testData = test.drop(['tracking_id','datetime','turbine_status'],axis=1)
    print(testData.shape)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X = sc.fit_transform(X)
    testData = sc.transform(testData)

    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=0)
    print(X_train.shape,y_train.shape)
    print(X_test.shape,y_test.shape)

    from sklearn.tree import DecisionTreeRegressor
    regressor_dt = DecisionTreeRegressor(random_state = 42)
    regressor_dt.fit(X_train, y_train)

    y_train_pred_dt = regressor_dt.predict(X_train)
    y_test_pred_dt = regressor_dt.predict(X_test)

    print(r2_score(y_true=y_train,y_pred=y_train_pred_dt))
    print(r2_score(y_true=y_test,y_pred=y_test_pred_dt))

    from sklearn.ensemble import RandomForestRegressor
    regressor_rf = RandomForestRegressor(n_estimators=200, n_jobs=1, oob_score=True, random_state=42)
    regressor_rf.fit(X_train, y_train)

    y_train_pred_rf = regressor_rf.predict(X_train)
    y_test_pred_rf = regressor_rf.predict(X_test)

    #print(r2_score(y_true=y_train,y_pred=y_train_pred_rf))
    #print(r2_score(y_true=y_test,y_pred=y_test_pred_rf))


    from xgboost import XGBRegressor
    regressor_xg = XGBRegressor(n_estimators=1000, max_depth=8, booster='gbtree', n_jobs=1, learning_rate=0.1, reg_lambda=0.01, reg_alpha=0.2)
    regressor_xg.fit(X_train, y_train)

    y_train_pred_xg = regressor_xg.predict(X_train)
    y_test_pred_xg = regressor_xg.predict(X_test)

    
    print(r2_score(y_true=y_train,y_pred=y_train_pred_xg))
    print(r2_score(y_true=y_test,y_pred=y_test_pred_xg))'''

    '''model = regressor_xg.predict(testData)
    model
    model.shape

    Ywrite=pd.DataFrame(model,columns=['windmill_generated_power(kW/h)'])
    var =pd.DataFrame(test[['tracking_id','datetime']])
    dataset_test_col = pd.concat([var,Ywrite], axis=1)
    dataset_test_col.to_csv("static/dataset/Prediction.csv",index=False)'''
    ff=open("static/det.txt","r")
    v=ff.read()
    vv=v.split(',')
    dataset = pd.read_csv('static/dataset/Prediction.csv')
    dat=dataset.head(200)

    data=[]
    for ss in dat.values:
        data.append(ss)
        
    return render_template('classify.html',vv=vv,data=data)


@app.route('/test_data', methods=['GET', 'POST'])
def test_data():
    act=""
    res=""
    uname=""
    if 'username' in session:
        uname = session['username']

    if uname is None:
        return redirect(url_for('login'))

    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM wt_register where uname=%s",(uname,))
    data = mycursor.fetchone()
    name=data[1]

    ws=""
    atmos_temp=""
    gear_temp=""
    eng_temp=""
    gen_temp=""
    atmos_pres=""
    area_temp=""
    wind_dir=""
    turbine_st=""
    cloud=""
    
    if request.method=='POST':
        ws=request.form['ws']
        atmos_temp=request.form['atmos_temp']
        gear_temp=request.form['gear_temp']
        eng_temp=request.form['eng_temp']
        gen_temp=request.form['gen_temp']
        atmos_pres=request.form['atmos_pres']
        area_temp=request.form['area_temp']
        wind_dir=request.form['wind_dir']
        turbine_st=request.form['turbine_st']
        cloud=request.form['cloud']
        

        df = pd.read_csv("static/dataset/train.csv")

        x=0

        ws1=float(ws)
        w1=ws1-3
        w2=ws1+3

        at1=float(atmos_temp)
        a1=at1-3
        a2=at1+3

        gr=float(gear_temp)
        gr1=gr-3
        gr2=gr+3

        eg=float(eng_temp)
        eg1=eg-3
        eg2=eg+3

        gn=float(gen_temp)
        gn1=gn-3
        gn2=gn+3

        ap=float(atmos_pres)
        ap1=ap-3
        ap2=ap+3

        at=float(area_temp)
        at1=at-3
        at2=at+3

        wd=float(wind_dir)
        wd1=wd-3
        wd2=wd+3

        
    
        for rr in df.values:
            if rr[2]>w1 and rr[2]<=w2 and rr[3]>a1 and rr[3]<=a2 and rr[6]>gr1 and rr[6]<=gr2 and rr[7]>eg1 and rr[7]<=eg2:
                print("a")
                if rr[9]>gn1 and rr[9]<gn2 and rr[10]>ap1 and rr[10]<=ap2 and rr[11]>at1 and rr[11]<=at2 and rr[13]>wd1 and rr[13]<=wd2:
                    print("b")
                    if rr[16]==turbine_st and rr[17]==cloud:
                        print("c")
                        res=rr[21]
                        x+=1
                        break

        print(x)
        
        if x>0:
            act="1"
        else:
            act="2"
                
            
    return render_template('test_data.html',name=name,res=res,act=act)




@app.route('/logout')
def logout():
    # remove the username from the session if it is there
    session.pop('username', None)
    return redirect(url_for('index'))



if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)


