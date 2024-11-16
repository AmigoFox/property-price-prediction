import pandas as pd
import numpy as np
from pandas import read_csv
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
pd.set_option('display.max_columns', None)
np.set_printoptions(suppress=True)

house = read_csv("C:/Users/lenovo/Desktop/AI_py/house.csv")
house.columns = house.columns.str.strip().str.lower()

price = float(input("Введи стоимость квартиры\n"))
area = float(input("Введи сколько кв м\n"))
rooms = int(input("Введи количество комнат\n"))
type = (input("Дом или квартира ?\n")).lower()
age = int(input("Введи дату застройки\n"))
floor = int(input("Введи этаж, на котором хочешь жить\n"))
parking = (input("Парковка нужна ?\n")).lower()
repair = (input("С ремонтом или нет ?\n")).lower()

if type == "дом":
    type = 1
else:
    type = 0

if parking == "да" or parking == "нужна":
    parking = 1
else:
    parking = 0

if repair == "да" or repair == "нужна":
    repair = 1
else:
    repair = 0

info = np.array([[area], [rooms], [type], [age], [floor], [parking], [repair]])

info_df = pd.DataFrame(info.T, columns=['площадь', 'количество комнат', 'тип недвижимости', 'год застройки', 'этаж', 'парковка', 'ремонт'])

X = house[['площадь', 'количество комнат', 'тип недвижимости', 'год застройки', 'этаж', 'парковка', 'ремонт']]
y = house['цена']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

info_scaled = scaler.transform(info_df)

model = LinearRegression()
model.fit(X_scaled, y)

predicted_price = model.predict(info_scaled)

print(f"Предсказанная цена: {predicted_price[0]}")