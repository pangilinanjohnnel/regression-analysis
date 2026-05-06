import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df=pd.read_csv(r"C:\Users\Johnnel\Desktop\TIP folder\free data\World Stock Prices ( Daily Updating )\archive\World-Stock-Prices-Dataset.csv")
df=df[df["Brand_Name"]=="apple"]

#fix
df["Date"] = pd.to_datetime(df["Date"], utc=True)

#SET&SORT
df.set_index("Date", inplace=True)
df.sort_index(inplace=True)

#DROP
df.drop(columns=["Capital Gains"], inplace=True)
df.drop_duplicates(inplace=True)

#train test
X = df[["Open", "High", "Low", "Volume"]]
y = df["Close"]

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)

#model
model=LinearRegression()
model.fit(X_train, y_train)

#prediction
y_pred=model.predict(X_test)

#scatter
plt.scatter(y_test, y_pred, c="r", label="Predicted")
plt.title("Scatter plot: Actual vs Predicted")
plt.xlabel("Apple:Open Price")
plt.ylabel("Apple:Close Price")
plt.legend()
plt.show()

