import pandas as pd# deals with data frame
import numpy as np# deals with numerical values

import matplotlib.pyplot as plt #for different types of plots
import statsmodels.formula.api as smf

from sklearn import datasets, linear_model, metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import seaborn as sns

dataset=pd.read_csv("C:/Users/Kumar Ambuj/PycharmProjects/DECISIONMINDS/venv/Cars.csv")
print(dataset)

print()
print('HP and WT')

plt.scatter(x=dataset['HP'],y=dataset['WT'],color='red')
model=smf.ols('WT~HP',data=dataset).fit()

pred = model.predict(pd.DataFrame(dataset['HP']))
plt.plot(pd.DataFrame(dataset['HP']),pred,color='black')
plt.xlabel('HP')
plt.ylabel('WT')
plt.show()

np.corrcoef(dataset['HP'],dataset['WT']) #correlation
model=smf.ols('WT~HP',data=dataset).fit()

print(model.params)
print(model.summary())
pred = model.predict(pd.DataFrame(dataset['HP']))
print(model.conf_int(0.05)) # 95% confidence interval


#MPG and Wt

print()
print('MPG and WT')

plt.scatter(x=dataset['MPG'],y=dataset['WT'],color='red')
model=smf.ols('WT~MPG',data=dataset).fit()

pred = model.predict(pd.DataFrame(dataset['MPG']))
plt.plot(pd.DataFrame(dataset['MPG']),pred,color='black')
plt.xlabel('MPG')
plt.ylabel('WT')
plt.show()

np.corrcoef(dataset['MPG'],dataset['WT']) #correlation
model=smf.ols('WT~MPG',data=dataset).fit()

print(model.params)
print(model.summary())
pred = model.predict(pd.DataFrame(dataset['MPG']))
print(model.conf_int(0.05)) # 95% confidence interval

# VOL and WT

print()
print('VOL and WT')

plt.scatter(x=dataset['VOL'],y=dataset['WT'],color='red')
model=smf.ols('WT~VOL',data=dataset).fit()

pred = model.predict(pd.DataFrame(dataset['VOL']))
plt.plot(pd.DataFrame(dataset['VOL']),pred,color='black')
plt.xlabel('VOL')
plt.ylabel('WT')
plt.show()

np.corrcoef(dataset['VOL'],dataset['WT']) #correlation
model=smf.ols('WT~VOL',data=dataset).fit()

print(model.params)
print(model.summary())
pred = model.predict(pd.DataFrame(dataset['VOL']))
print(model.conf_int(0.05)) # 95% confidence interval


#sp and wt

print()
print('SP and WT')

plt.scatter(x=dataset['SP'],y=dataset['WT'],color='red')
model=smf.ols('WT~SP',data=dataset).fit()

pred = model.predict(pd.DataFrame(dataset['SP']))
plt.plot(pd.DataFrame(dataset['SP']),pred,color='black')
plt.xlabel('SP')
plt.ylabel('WT')
plt.show()

np.corrcoef(dataset['SP'],dataset['WT']) #correlation
model=smf.ols('WT~SP',data=dataset).fit()

print(model.params)
print(model.summary())
pred = model.predict(pd.DataFrame(dataset['SP']))
print(model.conf_int(0.05)) # 95% confidence interval


#HP MPG and  WT

print('HP MPG and WT')
feature_cols = ['HP','MPG']


x=dataset[feature_cols]

y=dataset['WT']
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


#HP VOL and WT
print()
print()
print('HP VOL AND WT')
feature_cols = ['HP','VOL']


x=dataset[feature_cols]

y=dataset['WT']
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



#HP SP and Wt

print('HP SP and WT')
feature_cols = ['HP','SP']


x=dataset[feature_cols]

y=dataset['WT']
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

#  MPG VOL and WT
print()
print()
print('MPG VOL and WT')
feature_cols = ['MPG','VOL']


x=dataset[feature_cols]

y=dataset['WT']
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


# MPG SP and WT
print()
print()

print('MPG SP  and WT')
feature_cols = ['MPG','SP']


x=dataset[feature_cols]

y=dataset['WT']
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


# VOL SP and WT
print()
print()
print('VOL SP and WT')
feature_cols = ['VOL','SP']


x=dataset[feature_cols]

y=dataset['WT']
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



#HP MPG VOL and WT

print('HP MPG VOL and WT')
feature_cols = ['HP','MPG','VOL']


x=dataset[feature_cols]

y=dataset['WT']
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



#HP MPG SP and WT

print('HP MPG SP and WT')
feature_cols = ['HP','MPG','SP']


x=dataset[feature_cols]

y=dataset['WT']
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


#MPG VOL SP and WT
print()
print()

print('MPG VOL SP and WT')
feature_cols = ['MPG','VOL','SP']


x=dataset[feature_cols]

y=dataset['WT']
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


# HP MPG VOL SP and WT

print('HP MPG VOL SP and WT')
feature_cols = ['HP','MPG','VOL','SP',]


x=dataset[feature_cols]

y=dataset['WT']
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


#correlation inbetween variables

print()
print('HP and MPG')

plt.scatter(x=dataset['HP'],y=dataset['MPG'],color='red')
model=smf.ols('MPG~HP',data=dataset).fit()

pred = model.predict(pd.DataFrame(dataset['HP']))
plt.plot(pd.DataFrame(dataset['HP']),pred,color='black')
plt.xlabel('HP')
plt.ylabel('MPG')
plt.show()


print()
print('HP and VOL')

plt.scatter(x=dataset['HP'],y=dataset['VOL'],color='red')
model=smf.ols('VOL~HP',data=dataset).fit()

pred = model.predict(pd.DataFrame(dataset['HP']))
plt.plot(pd.DataFrame(dataset['HP']),pred,color='black')
plt.xlabel('HP')
plt.ylabel('VOL')
plt.show()


print()
print('HP and SP')

plt.scatter(x=dataset['HP'],y=dataset['SP'],color='red')
model=smf.ols('SP~HP',data=dataset).fit()

pred = model.predict(pd.DataFrame(dataset['HP']))
plt.plot(pd.DataFrame(dataset['HP']),pred,color='black')
plt.xlabel('HP')
plt.ylabel('SP')
plt.show()


print()
print('MPG and VOL')

plt.scatter(x=dataset['MPG'],y=dataset['VOL'],color='red')
model=smf.ols('VOL~MPG',data=dataset).fit()

pred = model.predict(pd.DataFrame(dataset['MPG']))
plt.plot(pd.DataFrame(dataset['MPG']),pred,color='black')
plt.xlabel('MPG')
plt.ylabel('VOL')
plt.show()

print()
print('MPG and SP')

plt.scatter(x=dataset['MPG'],y=dataset['SP'],color='red')
model=smf.ols('SP~MPG',data=dataset).fit()

pred = model.predict(pd.DataFrame(dataset['MPG']))
plt.plot(pd.DataFrame(dataset['MPG']),pred,color='black')
plt.xlabel('MPG')
plt.ylabel('SP')
plt.show()


print()
print('VOL and SP')

plt.scatter(x=dataset['VOL'],y=dataset['SP'],color='red')
model=smf.ols('SP~VOL',data=dataset).fit()

pred = model.predict(pd.DataFrame(dataset['VOL']))
plt.plot(pd.DataFrame(dataset['VOL']),pred,color='black')
plt.xlabel('VOL')
plt.ylabel('SP')
plt.show()