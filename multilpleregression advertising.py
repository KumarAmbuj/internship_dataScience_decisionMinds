import pandas as pd# deals with data frame
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd# deals with data frame
import numpy as np# deals with numerical values

import matplotlib.pyplot as plt #for different types of plots
import statsmodels.formula.api as smf

from sklearn import datasets, linear_model, metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import seaborn as sns

dataset=pd.read_csv("C:/Users/Kumar Ambuj/PycharmProjects/DECISIONMINDS/venv/Advertising.csv")
print(dataset)

print()
print('tv and sales')

plt.scatter(x=dataset['TV'],y=dataset['sales'],color='red')
model=smf.ols('sales~TV',data=dataset).fit()

pred = model.predict(pd.DataFrame(dataset['TV']))
plt.plot(pd.DataFrame(dataset['TV']),pred,color='black')
plt.xlabel('TV')
plt.ylabel('sales')
plt.show()

np.corrcoef(dataset['TV'],dataset['sales']) #correlation
model=smf.ols('sales~TV',data=dataset).fit()

print(model.params)
print(model.summary())
pred = model.predict(pd.DataFrame(dataset['TV']))
print(model.conf_int(0.05)) # 95% confidence interval

#radio and sales
print()
print()
print(' radio and sales ')

plt.scatter(x=dataset['radio'],y=dataset['sales'],color='red')
model=smf.ols('sales~radio',data=dataset).fit()

pred = model.predict(pd.DataFrame(dataset['radio']))
plt.plot(pd.DataFrame(dataset['radio']),pred,color='black')
plt.xlabel('radio')
plt.ylabel('sales')
plt.show()

np.corrcoef(dataset['radio'],dataset['sales']) #correlation
model=smf.ols('sales~radio',data=dataset).fit()

print(model.params)
print(model.summary())
pred = model.predict(pd.DataFrame(dataset['radio']))
print(model.conf_int(0.05)) # 95% confidence interval

#newspaper and sales
print()
print()
print('NewsPaper And Sales')

plt.scatter(x=dataset['newspaper'],y=dataset['sales'],color='red')
model=smf.ols('sales~newspaper',data=dataset).fit()

pred = model.predict(pd.DataFrame(dataset['newspaper']))
plt.plot(pd.DataFrame(dataset['newspaper']),pred,color='black')
plt.xlabel('newspaper')
plt.ylabel('sales')
plt.show()

np.corrcoef(dataset['newspaper'],dataset['sales']) #correlation
model=smf.ols('sales~newspaper',data=dataset).fit()

print(model.params)
print(model.summary())
pred = model.predict(pd.DataFrame(dataset['newspaper']))
print(model.conf_int(0.05)) # 95% confidence interval

# TV Radio and sales

print('TV Radio and sales')
feature_cols = ['TV','radio']


x=dataset[feature_cols]

y=dataset['sales']
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


# tv newspaper and sales

print('TV newspaper and sales')
feature_cols = ['TV','newspaper']


x=dataset[feature_cols]

y=dataset['sales']
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


#radio newspaper and sales

print()
print()
print('radio newspaper and sales')

feature_cols = ['radio','newspaper']


x=dataset[feature_cols]

y=dataset['sales']
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


#

