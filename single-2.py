import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
# import all libraries and methods

# importing data
data = pd.read_csv("used_car_dataset.csv")
df1 = pd.DataFrame(data)
df = pd.get_dummies(df1)
df['odometer'] = df['odometer'].fillna(df['odometer'].mean())

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

# data cleaning with price, year, and F2 with inter quartile rule
def single(a, b):

    X= df[list(df.columns)[a:b]]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    x = sm.add_constant(X_train)
    results = sm.OLS(y_train, x).fit()
    print(results.summary())

single(1, 2)
single(2, 3)
single(3, 4)
single(4, 5)
single(5, 6)
single(6, 8)
single(8, 12)
single(12, 15)
single(15, 16)
single(16, 18)
single(18, 22)
single(22, 27)
single(27, 30)