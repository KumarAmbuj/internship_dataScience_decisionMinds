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

dataset=pd.read_csv("C:/Users/Kumar Ambuj/PycharmProjects/DECISIONMINDS/Labour.csv")
#print(dataset)

# capital and wage

print('capital and wage')
plt.scatter(x=dataset['capital'],y=dataset['wage'],color='red')
model=smf.ols('wage~capital',data=dataset).fit()

pred = model.predict(pd.DataFrame(dataset['capital']))
plt.plot(pd.DataFrame(dataset['capital']),pred,color='black')
plt.xlabel('capital')
plt.ylabel('wage')
#plt.show()

np.corrcoef(dataset['capital'],dataset['wage']) #correlation
model=smf.ols('wage~capital',data=dataset).fit()

print(model.params)
print(model.summary())
pred = model.predict(pd.DataFrame(dataset['capital']))
print(model.conf_int(0.05)) # 95% confidence interval

#labour and wage

print('labour and wage')
plt.scatter(x=dataset['labour'],y=dataset['wage'],color='red')
model=smf.ols('wage~labour',data=dataset).fit()

pred = model.predict(pd.DataFrame(dataset['labour']))
plt.plot(pd.DataFrame(dataset['labour']),pred,color='black')
plt.xlabel('labour')
plt.ylabel('wage')
plt.show()

np.corrcoef(dataset['labour'],dataset['wage']) #correlation
model=smf.ols('wage~labour',data=dataset).fit()

print(model.params)
print(model.summary())
pred = model.predict(pd.DataFrame(dataset['labour']))
print(model.conf_int(0.05)) # 95% confidence interval


#outut and wage


print('output and wage')

plt.scatter(x=dataset['output'],y=dataset['wage'],color='red')
model=smf.ols('wage~output',data=dataset).fit()

pred = model.predict(pd.DataFrame(dataset['output']))
plt.plot(pd.DataFrame(dataset['output']),pred,color='black')
plt.xlabel('output')
plt.ylabel('wage')
#plt.show()

np.corrcoef(dataset['output'],dataset['wage']) #correlation
model=smf.ols('wage~output',data=dataset).fit()

print(model.params)
print(model.summary())
pred = model.predict(pd.DataFrame(dataset['output']))
print(model.conf_int(0.05)) # 95% confidence interval


#capital labour and wage
print('capital labour and wage')
feature_cols = ['capital','labour']


x=dataset[feature_cols]
print(x)
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

print('mean absolute error:',metrics.mean_absolute_error(y_test,y_pred))
print('mean square error:',metrics.mean_squared_error(y_test,y_pred))
print(np.sqrt(metrics.mean_squared_error(y_test,y_test)))


#capital output and wage

print('capital outpiut and wage')

feature_cols = ['capital','output']


x=dataset[feature_cols]
print(x)
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

print('mean absolute error:',metrics.mean_absolute_error(y_test,y_pred))
print('mean square error:',metrics.mean_squared_error(y_test,y_pred))
print(np.sqrt(metrics.mean_squared_error(y_test,y_test)))


#labour ouptut and wage

print('labour output and wage')
feature_cols = ['labour','output']


x=dataset[feature_cols]
print(x)
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

print('mean absolute error:',metrics.mean_absolute_error(y_test,y_pred))
print('mean square error:',metrics.mean_squared_error(y_test,y_pred))
print(np.sqrt(metrics.mean_squared_error(y_test,y_test)))


#checking for collinearity

#graphs between independent variables
#capital and labour
plt.scatter(x=dataset['capital'],y=dataset['labour'],color='red')
model=smf.ols('labour~capital',data=dataset).fit()

pred = model.predict(pd.DataFrame(dataset['capital']))
plt.plot(pd.DataFrame(dataset['capital']),pred,color='black')
plt.xlabel('capital')
plt.ylabel('labour')
plt.show()

np.corrcoef(dataset['capital'],dataset['labour']) #correlation
model=smf.ols('labour~capital',data=dataset).fit()

print(model.params)
print(model.summary())
pred = model.predict(pd.DataFrame(dataset['capital']))
print(model.conf_int(0.05)) # 95% confidence interval

#capital and output

plt.scatter(x=dataset['capital'],y=dataset['output'],color='red')
model=smf.ols('output~capital',data=dataset).fit()

pred = model.predict(pd.DataFrame(dataset['capital']))
plt.plot(pd.DataFrame(dataset['capital']),pred,color='black')
plt.xlabel('capital')
plt.ylabel('output')
plt.show()

np.corrcoef(dataset['capital'],dataset['output']) #correlation
model=smf.ols('labour~capital',data=dataset).fit()

print(model.params)
print(model.summary())
pred = model.predict(pd.DataFrame(dataset['capital']))
print(model.conf_int(0.05)) # 95% confidence interval

#labour and output

plt.scatter(x=dataset['labour'],y=dataset['output'],color='red')
model=smf.ols('output~labour',data=dataset).fit()

pred = model.predict(pd.DataFrame(dataset['labour']))
plt.plot(pd.DataFrame(dataset['labour']),pred,color='black')
plt.xlabel('labour')
plt.ylabel('output')
plt.show()

np.corrcoef(dataset['labour'],dataset['output']) #correlation
model=smf.ols('output~labour',data=dataset).fit()

print(model.params)
print(model.summary())
pred = model.predict(pd.DataFrame(dataset['labour']))
print(model.conf_int(0.05)) # 95% confidence interval


#best fit model output and wage
print()
print()
print()
print('output and wage')

plt.scatter(x=dataset['output'],y=dataset['wage'],color='red')
model=smf.ols('wage~output',data=dataset).fit()

pred = model.predict(pd.DataFrame(dataset['output']))
plt.plot(pd.DataFrame(dataset['output']),pred,color='black')
plt.xlabel('output')
plt.ylabel('wage')
plt.show()

np.corrcoef(dataset['output'],dataset['wage']) #correlation
model=smf.ols('wage~output',data=dataset).fit()

print(model.params)
print(model.summary())
pred = model.predict(pd.DataFrame(dataset['output']))
print(model.conf_int(0.05)) # 95% confidence interval







