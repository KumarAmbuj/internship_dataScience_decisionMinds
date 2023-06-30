import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



dataset=pd.read_csv("C:/Users/Kumar Ambuj/PycharmProjects/DECISIONMINDS/wc-at.csv")
print(dataset)

waist=dataset.iloc[:,0]
at=dataset.iloc[:,1]

print(waist)
print(at)


plt.scatter(waist, at, color='green')
plt.xlabel('waist')
plt.ylabel('at')
plt.title('waist vs at graph')
plt.show()

from sklearn.model_selection import train_test_split
waist_train, waist_test, at_train, at_test=train_test_split(waist, at, test_size=0.2, random_state=0)

print(waist_train)
print(waist_test)
print(at_train)
print(at_test)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(waist_train, at_train)







