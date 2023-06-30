import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

claimants = pd.read_csv("C:/Users/Kumar Ambuj/PycharmProjects/DECISIONMINDS/venv/claimants.csv")

print(claimants.head(10))

claimants = claimants.apply(lambda x:x.fillna(x.value_counts().index[0]))

x = claimants.drop(["ATTORNEY"],axis=1)
y = claimants["ATTORNEY"]


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.3,random_state=1)

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

print(logmodel.fit(x_train,y_train))

predictions = logmodel.predict(x_test)

print(predictions)

from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,predictions))