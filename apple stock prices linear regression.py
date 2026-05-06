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

#check
df.info()
print(" ")

print("dimension")
print(df.shape)
print(" ")

print("null")
print(df.isna().sum())
print(" ")

print("dups")
print(df.duplicated().sum())
print(" ")

print(df.head())
print(df.tail())

#fix
df["Date"] = pd.to_datetime(df["Date"], utc=True)

#SET&SORT
df.set_index("Date", inplace=True)
df.sort_index(inplace=True)

#DROP
df.drop(columns=["Capital Gains"], inplace=True)
df.drop_duplicates(inplace=True)

#train test
X = np.array(df["Open"]).reshape(-1,1)
y = np.array(df["Close"]).reshape(-1,1)

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)

#model
model=LinearRegression()
model.fit(X_train, y_train)

#prediction
y_pred=model.predict(X_test)

#scatter
plt.scatter(X_test, y_test, c="b", label="Actual")
plt.scatter(X_test, y_pred, c="r", label="Predicted")
plt.title("Scatter plot: Actual vs Predicted")
plt.xlabel("Apple:Open Price")
plt.ylabel("Apple:Close Price")
plt.legend()
plt.show()

#regression
plt.scatter(X_test, y_test, c="b", label="Actual")
plt.plot(X_test, y_pred, c="r", label="Regression line", lw=3)
plt.title("Regression Plot: Actual vs Predicted")
plt.xlabel("Apple:Open Price")
plt.ylabel("Apple:Close Price")
plt.legend()
plt.show()

#heatmap
corr = df[["Open", "Close"]].corr()
plt.figure(figsize=(4,3))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Heatmap: Open vs Close")
plt.show()


#evaluation
mse=mean_squared_error(y_test, y_pred)
rmse=np.sqrt(mse)
mae=mean_absolute_error(y_test, y_pred)
r2=model.score(X_test, y_test)

print(f"MSE:{mse:.4f}")
print(f"RMSE{rmse:.4f}")
print(f"MAE:{mae:.4f}")
print(f"R2:{r2:.4f}")

