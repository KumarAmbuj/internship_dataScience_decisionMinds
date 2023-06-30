import pandas as pd# deals with data frame
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd# deals with data frame
import numpy as np# deals with numerical values
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt #for different types of plots
import statsmodels.formula.api as smf

dataset=pd.read_csv("C:/Users/Kumar Ambuj/PycharmProjects/DECISIONMINDS/Labour.csv")
print(dataset)

df=pd.DataFrame(dataset)
print(df)

x=df[['capital','labour']]
print(x)

dataset.head()
sns.pairplot(dataset,x_vars=['capital','labour','output'],y_vars='wage',kind='reg')

feature_cols = ['cspital','labour','output']
x=dataset[feature_cols]
y=dataset['wage']
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
print(y_pred)


