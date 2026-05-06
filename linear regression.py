import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

df = pd.read_csv(r"C:\Users\Johnnel\Desktop\TIP folder\1st year 2nd sem\applied machine learning\to do\ex 2\Exam score prediction\Exam_Score_Prediction.csv")

#Correlation Analysis (Choose Highest)
numeric_df = df.select_dtypes(include=np.number)
corr = numeric_df.corr()["exam_score"].sort_values(ascending=False)
print(corr)

#UNIVARIATE REGRESSION
X = df[["study_hours"]]
y = df["exam_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_uni = LinearRegression()
model_uni.fit(X_train, y_train)

y_pred = model_uni.predict(X_test)

#Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = model_uni.score(X_test, y_test)

print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r2)

#MULTIVARIATE REGRESSION
X_multi = df[['study_hours', 'class_attendance', 'sleep_hours']]
X_multi = sm.add_constant(X_multi)

y = df["exam_score"]

model = sm.OLS(y, X_multi).fit()
model.summary()





