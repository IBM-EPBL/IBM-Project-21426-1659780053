import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

from sklearn.metrics import r2_score
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

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

X = train.drop(['tracking_id','datetime','windmill_generated_power(kW/h)','turbine_status'],axis=1)
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

print(r2_score(y_true=y_train,y_pred=y_train_pred_rf))
print(r2_score(y_true=y_test,y_pred=y_test_pred_rf))


from xgboost import XGBRegressor
regressor_xg = XGBRegressor(n_estimators=1000, max_depth=8, booster='gbtree', n_jobs=1, learning_rate=0.1, reg_lambda=0.01, reg_alpha=0.2)
regressor_xg.fit(X_train, y_train)

y_train_pred_xg = regressor_xg.predict(X_train)
y_test_pred_xg = regressor_xg.predict(X_test)

print(r2_score(y_true=y_train,y_pred=y_train_pred_xg))
print(r2_score(y_true=y_test,y_pred=y_test_pred_xg))

model = regressor_xg.predict(testData)
model
model.shape

Ywrite=pd.DataFrame(model,columns=['windmill_generated_power(kW/h)'])
var =pd.DataFrame(test[['tracking_id','datetime']])
dataset_test_col = pd.concat([var,Ywrite], axis=1)
dataset_test_col.to_csv("static/dataset/Prediction.csv",index=False)















