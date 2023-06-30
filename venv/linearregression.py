# For reading data set
# importing necessary libraries
import pandas as pd# deals with data frame
import numpy as np# deals with numerical values
wcat=pd.read_csv("C:/Users/Kumar Ambuj/PycharmProjects/DECISIONMINDS/wc-at.csv")
#print(wcat)
import matplotlib.pyplot as plt #for different types of plots
import statsmodels.formula.api as smf

print(wcat['Waist'])

plt.scatter(x=wcat['Waist'],y=wcat['AT'],color='red')
model=smf.ols('AT~Waist',data=wcat).fit()
pred = model.predict(pd.DataFrame(wcat['Waist']))
plt.plot(pd.DataFrame(wcat['Waist']),pred,color='black')
plt.xlabel('WAIST')
plt.ylabel('TISSUE')
plt.show()
np.corrcoef(wcat['Waist'],wcat['AT']) #correlation
model=smf.ols('AT~Waist',data=wcat).fit()

print(model.params)
print(model.summary())
pred = model.predict(pd.DataFrame(wcat['Waist']))
print(model.conf_int(0.05)) # 95% confidence interval

# Transforming variables
model2 = smf.ols('AT~np.log(Waist)',data=wcat).fit()
model2.params
model2.summary()
pred2 = model2.predict(pd.DataFrame(wcat['Waist']))
#print(pred2)
#print(model2.conf_int(0.01)) # 99% confidence level
plt.scatter(x=wcat['Waist'],y=wcat['AT'],color='green')
plt.plot(pd.DataFrame(wcat['Waist']),pred2,color='blue')
plt.xlabel('WAIST')
plt.ylabel('TISSUE')
#plt.show()
# Exponential transformation
model3 = smf.ols('np.log(AT)~Waist',data=wcat).fit()
model3.params
model3.summary()
pred_log = model3.predict(pd.DataFrame(wcat['Waist']))
pred_log
pred3=np.exp(pred_log)
#print(pred3)
#print(model3.conf_int(0.01)) # 99% confidence level
#plt.scatter(x=wcat['Waist'],y=wcat['AT'],color='green')
plt.plot(pd.DataFrame(wcat['Waist']),pred3,color='blue')
plt.xlabel('WAIST')
plt.ylabel('TISSUE')
#plt.show()