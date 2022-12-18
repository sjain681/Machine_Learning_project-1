import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# import all libraries and methods

# importing data
data = pd.read_csv("used_car_dataset.csv")
df1 = pd.DataFrame(data)
df = pd.get_dummies(df1) #one hot encoding
df['odometer'] = df['odometer'].fillna(df['odometer'].mean()) # adding mean values wherever NaN

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

# regression with each variable (sklearn)
def single(a, b):

    X= df[list(df.columns)[a:b]]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    model = LinearRegression()
    model.fit(X_train, y_train)
    print('score:', model.score(X_test,y_test))
    print('coefficient:',  model.coef_)
    print('intercept:', model.intercept_)

print('year: '), single(1, 2)
print('odometer'), single(2, 3)
print('F1'), single(3, 4)
print('F2'), single(4, 5)
print('F3'), single(5, 6)
print('manufacturer'), single(6, 8)
print('condition'), single(8, 12)
print('cylinders'), single(12, 15)
print('fuel'), single(15, 16)
print('transmission'), single(16, 18)
print('type'), single(18, 22)
print('paint_color'), single(22, 27)
print('F4'), single(27, 30)

