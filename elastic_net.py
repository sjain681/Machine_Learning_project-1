import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import metrics
# import all libraries and methods

# importing data
data = pd.read_csv("used_car_dataset.csv")
df1 = pd.DataFrame(data)
df = pd.get_dummies(df1) #one hot encoding
df['odometer'] = df['odometer'].fillna(df['odometer'].mean()) # adding mean values wherever NaN

# VIF
X = df[['price', 'year', 'odometer', 'F1', 'F2', 'F3']]
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                   for i in range(len(X.columns))]
print(vif_data)

# data cleaning with price, year, and F2 with inter quartile rule
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
fil_ter = (df['price'] >= Q1 - 1.5*IQR) & (df['price'] <= Q3 + 1.5*IQR)
df = df.loc[fil_ter]

q1 = df['year'].quantile(0.25)
q3 = df['year'].quantile(0.75)
IqR = q3 - q1
fi_lter = (df['year'] >= q1 - 1.5*IqR) & (df['year'] <= q3 + 1.5*IqR)
df = df.loc[fi_lter]

k1 = df['F2'].quantile(0.25)
k3 = df['F2'].quantile(0.75)
IkR = k3 - k1
f_ilter = (df['F2'] >= k1 - 1.5*IkR) & (df['F2'] <= k3 + 1.5*IkR)
df = df.loc[f_ilter]

#splitting data
X= df[list(df.columns)[1:]]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

#Elastic Net
alphas = 10**np.linspace(10,-2,100)*0.5
elastic_cv= linear_model.ElasticNetCV(alphas=alphas, cv=5)
elastic = elastic_cv.fit(X_train, y_train)
EN = linear_model.ElasticNet(alpha = elastic.alpha_)
EN.fit(X_train, y_train)
y_pred = EN.predict(X_test)
print("ElasticNet regression R-Squared: ",metrics.r2_score(y_test,y_pred))
print("ElasticNet regression RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))