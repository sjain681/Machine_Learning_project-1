import pandas as pd
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
#import category_encoders as ce


def one_hot_encoding(data):

    values = array(data)
    print(values)

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    print(integer_encoded)

    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    print(onehot_encoded)

    # invert first example
    inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
    print(inverted)

data = pd.read_csv("used_car_dataset.csv")
df = pd.DataFrame(data)

#y = pd.get_dummies(df)

manufacturer = df['manufacturer']
# condition = df['condition']
# cylinders = df['cylinders']
# fuel = df['fuel']
# transmission = df['transmission']
# type = df['type']
# paint_color = df['paint_color']
# F4 = df['F4']
#
one_hot_encoding(manufacturer)
# one_hot_encoding(condition)
# one_hot_encoding(cylinders)
# one_hot_encoding(fuel)
# one_hot_encoding(transmission)
# one_hot_encoding(type)
# one_hot_encoding(paint_color)
# one_hot_encoding(F4)