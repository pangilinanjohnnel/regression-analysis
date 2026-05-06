import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

train = pd.read_csv(r"C:\Users\Johnnel\Desktop\TIP folder\1st year 2nd sem\applied machine learning\to do\D10\archive\train_energy_data.csv")
test = pd.read_csv(r"C:\Users\Johnnel\Desktop\TIP folder\1st year 2nd sem\applied machine learning\to do\D10\archive\test_energy_data.csv")

X_train = train.drop(columns="Energy Consumption")
y_train = train["Energy Consumption"]

X_test = test.drop(columns="Energy Consumption")
y_test = test["Energy Consumption"]

# Encode
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

#Linear Regression
LR = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

LR.fit(X_train, y_train)

LR_pred = LR.predict(X_test)

# metrics
mse = metrics.mean_squared_error(y_test, LR_pred)
rmse = np.sqrt(mse)
mae = metrics.mean_absolute_error(y_test, LR_pred)
mape = metrics.mean_absolute_percentage_error(y_test, LR_pred)
r2 = metrics.r2_score(y_test, LR_pred)

print("LIN REG")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.4f}")
print(f"R2 Score: {r2:.4f}")

#RIDGE
ridge = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge())
])

ridge_param = {
    "ridge__alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    "ridge__fit_intercept": [True, False],
    "ridge__solver": ["auto", "svd", "cholesky", "lsqr"]}

GR = GridSearchCV(
    ridge,
    ridge_param,
    cv=5,
    scoring="neg_mean_squared_error")

GR.fit(X_train, y_train)
GR_pred = GR.predict(X_test)

# metrics
mse = metrics.mean_squared_error(y_test, GR_pred)
rmse = np.sqrt(mse)
mae = metrics.mean_absolute_error(y_test, GR_pred)
mape = metrics.mean_absolute_percentage_error(y_test, GR_pred)
r2 = metrics.r2_score(y_test, GR_pred)

print("RIDGE")
print(GR.best_params_)
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.4f}")
print(f"R2 Score: {r2:.4f}")

#Decision Tree
dt = Pipeline([
    ("model", DecisionTreeRegressor(random_state=42))])

dt_param = {
    "model__max_depth": [3, 5, 10, 15, None],
    "model__min_samples_split": [2, 5, 10, 20],
    "model__min_samples_leaf": [1, 2, 5, 10],
    "model__max_features": [None, "sqrt", "log2"]}

GD= GridSearchCV(
    dt,
    dt_param,
    cv=5,
    scoring="neg_mean_squared_error")

GD.fit(X_train, y_train)
DT_pred = GD.predict(X_test)

# metrics
mse = metrics.mean_squared_error(y_test, DT_pred)
rmse = np.sqrt(mse)
mae = metrics.mean_absolute_error(y_test, DT_pred)
mape = metrics.mean_absolute_percentage_error(y_test, DT_pred)
r2 = metrics.r2_score(y_test, DT_pred)

print("DEC TREE")
print(GD.best_params_)
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.4f}")
print(f"R2 Score: {r2:.4f}")