import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import math
from pandas.plotting import scatter_matrix
# import all libraries and methods

# importing data
data = pd.read_csv("used_car_dataset.csv")
df1 = pd.DataFrame(data)
df = pd.get_dummies(df1)
df['odometer'] = df['odometer'].fillna(df['odometer'].mean())

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
"""
#correlations heatmap
corr = df1.corr()# plot the heatmap
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))
plt.title("heatmap")

# boxplot with cleaned data
fig1 = plt.figure()
g = sns.boxplot(x='year', y='price', data=df)
g.set_title('year vs Price')
plt.xticks(rotation=45)

# boxplot before cleaning data
fig2 = plt.figure()
g1 = sns.boxplot(x='year', y='price', data=df1)
g1.set_title('year vs Price')
plt.xticks(rotation=90)

# scatter matrix plot
attributes = ["price", "odometer", "year", "F1", "F2", "F3"]
scatter_matrix(df1[attributes])
plt.title("scatter matrix plot")

# scatter plot price vs. year
df.plot(kind="scatter", x="year", y="price")
plt.title("price vs. year")

# scatter plot year vs. F2
df1.plot(kind="scatter", x="F2", y="year")
plt.title("year vs. F2")

#histogram price
plt.figure()
df["price"].hist(bins = 10, grid = False)
plt.title("price histogram")

# histogram year
plt.figure()
df["year"].hist(bins = 25, grid = False)
plt.title("year histogram")

# histogram F2
plt.figure()
df["F2"].hist(bins = 25, grid = False)
plt.title("F2 histogram")
"""
#histogram F3
plt.figure()
df["F3"].hist(bins = 40, grid = False)
plt.title("F3 histogram")

plt.show()