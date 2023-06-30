import pandas as pd# deals with data frame
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf
import seaborn as sns

adv=pd.read_csv("C:/Users/Kumar Ambuj/PycharmProjects/DECISIONMINDS/venv/Advertising.csv")

adv.head()
sns.pairplot(adv,x_vars=['TV','radio','newspaper'],y_vars='sales',kind='reg')

feature_cols = ['TV','radio']


x=adv[feature_cols]
print(x)
y=adv['sales']
x_train, x_test, y_train, y_test = train_test_split(x,y)
linreg = LinearRegression()
linreg.fit(x_train,y_train)

print('coefficient of determination r-square:', linreg.score(x_train,y_train))
print('intercept:', linreg.intercept_)
print('slope:', linreg.coef_)
print('Fit intercept:',linreg.fit_intercept)
z = zip(feature_cols,linreg.coef_)

print('feature coefficients:',z)

y_pred = linreg.predict(x_test)

print('mean absolute error:',metrics.mean_absolute_error(y_test,y_pred))
print('mean square error:',metrics.mean_squared_error(y_test,y_pred))
print(np.sqrt(metrics.mean_squared_error(y_test,y_test)))




