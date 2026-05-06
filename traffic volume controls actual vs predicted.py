import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df=pd.read_csv(r"C:\Users\Johnnel\Desktop\TIP folder\1st year 2nd sem\applied machine learning\A2\metro+interstate+traffic+volume\Metro_Interstate_Traffic_Volume.csv\Metro_Interstate_Traffic_Volume.csv")

df["date_time"] = pd.to_datetime(df["date_time"], utc=True)
df["hour"] = df["date_time"].dt.hour

df["is_holiday"] = df["holiday"].notna().astype(int)
df.drop(columns=['holiday'], inplace=True)

#HOURS only
X = np.array(df["hour"]).reshape(-1,1)
y = np.array(df["traffic_volume"]).reshape(-1,1)

poly = PolynomialFeatures(degree=2, include_bias=False)

X_poly = poly.fit_transform(X)

X_train, X_test, y_train, y_test=train_test_split(X_poly, y, test_size=0.2, random_state=42)

#model
model=LinearRegression()
model.fit(X_train, y_train)

#prediction
y_pred=model.predict(X_test)

#scatter
plt.scatter(y_test, y_pred, c="r", label="Predicted")
plt.title("Actual vs Predicted Hour only")
plt.xlabel("Actual Traffic Volume")
plt.ylabel("Predicted Traffic Volume")
plt.legend()
plt.show()

#Evaluation
mse=mean_squared_error(y_test, y_pred)
rmse=np.sqrt(mse)
mae=mean_absolute_error(y_test, y_pred)
r2=model.score(X_test, y_test)

print("Evaluation Metrics HOURS only")
print(f"MSE:{mse:.4f}")
print(f"RMSE{rmse:.4f}")
print(f"MAE:{mae:.4f}")
print(f"R2:{r2:.4f}")
print(" ")

#REGULAR DAY model
X = df[["hour", "temp"]]
y = df["traffic_volume"]

poly = PolynomialFeatures(degree=2, include_bias=False)

X_poly = poly.fit_transform(X)

X_train, X_test, y_train, y_test=train_test_split(X_poly, y, test_size=0.2, random_state=42)

#model
model=LinearRegression()
model.fit(X_train, y_train)

#prediction
y_pred=model.predict(X_test)

#scatter
plt.scatter(y_test, y_pred, c="b", label="Predicted")
plt.title("REGULAR DAY Actual vs Predicted")
plt.xlabel("Actual Traffic Volume")
plt.ylabel("Predicted Traffic Volume")
plt.legend()
plt.show()

#Evaluation
mse=mean_squared_error(y_test, y_pred)
rmse=np.sqrt(mse)
mae=mean_absolute_error(y_test, y_pred)
r2=model.score(X_test, y_test)

print("Evaluation Metrics REGULAR DAY")
print(f"MSE:{mse:.4f}")
print(f"RMSE{rmse:.4f}")
print(f"MAE:{mae:.4f}")
print(f"R2:{r2:.4f}")
print(" ")

#HOLIDAY
X = df[["hour", "temp", "is_holiday"]]
y = df["traffic_volume"]

poly = PolynomialFeatures(degree=2, include_bias=False)

X_poly = poly.fit_transform(X)

X_train, X_test, y_train, y_test=train_test_split(X_poly, y, test_size=0.2, random_state=42)

#model
model=LinearRegression()
model.fit(X_train, y_train)

#prediction
y_pred=model.predict(X_test)

#scatter
plt.scatter(y_test, y_pred, c="g", label="Predicted")
plt.title("HOLIDAY Actual vs Predicted")
plt.xlabel("Actual Traffic Volume")
plt.ylabel("Predicted Traffic Volume")
plt.legend()
plt.show()

#Evaluation
mse=mean_squared_error(y_test, y_pred)
rmse=np.sqrt(mse)
mae=mean_absolute_error(y_test, y_pred)
r2=model.score(X_test, y_test)

print("Evaluation Metrics HOLIDAY")
print(f"MSE:{mse:.4f}")
print(f"RMSE{rmse:.4f}")
print(f"MAE:{mae:.4f}")
print(f"R2:{r2:.4f}")
print(" ")

#rainy REGULAR DAY model
X = df[["hour", "temp", "rain_1h"]]
y = df["traffic_volume"]

poly = PolynomialFeatures(degree=2, include_bias=False)

X_poly = poly.fit_transform(X)

X_train, X_test, y_train, y_test=train_test_split(X_poly, y, test_size=0.2, random_state=42)

#model
model=LinearRegression()
model.fit(X_train, y_train)

#prediction
y_pred=model.predict(X_test)

#scatter
plt.scatter(y_test, y_pred, c="b", label="Predicted")
plt.title("rainy REGULAR DAY Actual vs Predicted")
plt.xlabel("Actual Traffic Volume")
plt.ylabel("Predicted Traffic Volume")
plt.legend()
plt.show()

#Evaluation
mse=mean_squared_error(y_test, y_pred)
rmse=np.sqrt(mse)
mae=mean_absolute_error(y_test, y_pred)
r2=model.score(X_test, y_test)

print("Evaluation Metrics rainy REGULAR DAY")
print(f"MSE:{mse:.4f}")
print(f"RMSE{rmse:.4f}")
print(f"MAE:{mae:.4f}")
print(f"R2:{r2:.4f}")
print(" ")

#rainy HOLIDAY
X = df[["hour", "temp", "is_holiday", "rain_1h"]]
y = df["traffic_volume"]

poly = PolynomialFeatures(degree=2, include_bias=False)

X_poly = poly.fit_transform(X)

X_train, X_test, y_train, y_test=train_test_split(X_poly, y, test_size=0.2, random_state=42)

#model
model=LinearRegression()
model.fit(X_train, y_train)

#prediction
y_pred=model.predict(X_test)

#scatter
plt.scatter(y_test, y_pred, c="g", label="Predicted")
plt.title("rainy HOLIDAY Actual vs Predicted")
plt.xlabel("Actual Traffic Volume")
plt.ylabel("Predicted Traffic Volume")
plt.legend()
plt.show()

#Evaluation
mse=mean_squared_error(y_test, y_pred)
rmse=np.sqrt(mse)
mae=mean_absolute_error(y_test, y_pred)
r2=model.score(X_test, y_test)

print("Evaluation Metrics rainy HOLIDAY")
print(f"MSE:{mse:.4f}")
print(f"RMSE{rmse:.4f}")
print(f"MAE:{mae:.4f}")
print(f"R2:{r2:.4f}")
print(" ")

#snowy REGULAR DAY model
X = df[["hour", "temp", "snow_1h"]]
y = df["traffic_volume"]

poly = PolynomialFeatures(degree=2, include_bias=False)

X_poly = poly.fit_transform(X)

X_train, X_test, y_train, y_test=train_test_split(X_poly, y, test_size=0.2, random_state=42)

#model
model=LinearRegression()
model.fit(X_train, y_train)

#prediction
y_pred=model.predict(X_test)

#scatter
plt.scatter(y_test, y_pred, c="b", label="Predicted")
plt.title("snowy REGULAR DAY Actual vs Predicted")
plt.xlabel("Actual Traffic Volume")
plt.ylabel("Predicted Traffic Volume")
plt.legend()
plt.show()

#Evaluation
mse=mean_squared_error(y_test, y_pred)
rmse=np.sqrt(mse)
mae=mean_absolute_error(y_test, y_pred)
r2=model.score(X_test, y_test)

print("Evaluation Metrics snowy REGULAR DAY")
print(f"MSE:{mse:.4f}")
print(f"RMSE{rmse:.4f}")
print(f"MAE:{mae:.4f}")
print(f"R2:{r2:.4f}")
print(" ")

#snowy HOLIDAY
X = df[["hour", "temp", "is_holiday", "snow_1h"]]
y = df["traffic_volume"]

poly = PolynomialFeatures(degree=2, include_bias=False)

X_poly = poly.fit_transform(X)

X_train, X_test, y_train, y_test=train_test_split(X_poly, y, test_size=0.2, random_state=42)

#model
model=LinearRegression()
model.fit(X_train, y_train)

#prediction
y_pred=model.predict(X_test)

#scatter
plt.scatter(y_test, y_pred, c="g", label="Predicted")
plt.title("snowy HOLIDAY Actual vs Predicted")
plt.xlabel("Actual Traffic Volume")
plt.ylabel("Predicted Traffic Volume")
plt.legend()
plt.show()

mse=mean_squared_error(y_test, y_pred)
rmse=np.sqrt(mse)
mae=mean_absolute_error(y_test, y_pred)
r2=model.score(X_test, y_test)

print("Evaluation Metrics snowy HOLIDAY")
print(f"MSE:{mse:.4f}")
print(f"RMSE{rmse:.4f}")
print(f"MAE:{mae:.4f}")
print(f"R2:{r2:.4f}")
print(" ")

#cloudy REGULAR DAY model
X = df[["hour", "temp", "clouds_all"]]
y = df["traffic_volume"]

poly = PolynomialFeatures(degree=2, include_bias=False)

X_poly = poly.fit_transform(X)

X_train, X_test, y_train, y_test=train_test_split(X_poly, y, test_size=0.2, random_state=42)

#model
model=LinearRegression()
model.fit(X_train, y_train)

#prediction
y_pred=model.predict(X_test)

#scatter
plt.scatter(y_test, y_pred, c="b", label="Predicted")
plt.title("cloudy REGULAR DAY Actual vs Predicted")
plt.xlabel("Actual Traffic Volume")
plt.ylabel("Predicted Traffic Volume")
plt.legend()
plt.show()

#Evaluation
mse=mean_squared_error(y_test, y_pred)
rmse=np.sqrt(mse)
mae=mean_absolute_error(y_test, y_pred)
r2=model.score(X_test, y_test)

print("Evaluation Metrics cloudy REGULAR DAY")
print(f"MSE:{mse:.4f}")
print(f"RMSE{rmse:.4f}")
print(f"MAE:{mae:.4f}")
print(f"R2:{r2:.4f}")
print(" ")

#cloudy HOLIDAY
X = df[["hour", "temp", "is_holiday", "clouds_all"]]
y = df["traffic_volume"]

poly = PolynomialFeatures(degree=2, include_bias=False)

X_poly = poly.fit_transform(X)

X_train, X_test, y_train, y_test=train_test_split(X_poly, y, test_size=0.2, random_state=42)

#model
model=LinearRegression()
model.fit(X_train, y_train)

#prediction
y_pred=model.predict(X_test)

#scatter
plt.scatter(y_test, y_pred, c="g", label="Predicted")
plt.title("cloudy HOLIDAY Actual vs Predicted")
plt.xlabel("Actual Traffic Volume")
plt.ylabel("Predicted Traffic Volume")
plt.legend()
plt.show()

#Evaluation
mse=mean_squared_error(y_test, y_pred)
rmse=np.sqrt(mse)
mae=mean_absolute_error(y_test, y_pred)
r2=model.score(X_test, y_test)

print("Evaluation Metrics cloudy HOLIDAY")
print(f"MSE:{mse:.4f}")
print(f"RMSE{rmse:.4f}")
print(f"MAE:{mae:.4f}")
print(f"R2:{r2:.4f}")
print(" ")
