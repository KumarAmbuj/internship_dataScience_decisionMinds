import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

claimants = pd.read_csv("C:/Users/Kumar Ambuj/PycharmProjects/DECISIONMINDS/venv/claimants.csv")

print(claimants.head(10))

print(claimants.info())

print(sns.countplot(x="ATTORNEY",data=claimants))

print(sns.countplot(x="ATTORNEY",hue="SEATBELT",data=claimants))

claimants["CLMAGE"].plot.hist()

claimants = claimants.drop(["CASENUM"],axis=1)

print(claimants.head(10))

#checking is data is null
print(claimants.isnull().sum())

sns.heatmap(claimants.isnull(), yticklabels=False,cmap="viridis")

claimants = claimants.apply(lambda x:x.fillna(x.value_counts().index[0]))

print(claimants.head(10))

print(claimants.isnull().sum())

import numpy as np
import statsmodels.formula.api as sm
logit_model = sm.logit('ATTORNEY~CLMAGE+LOSS+CLMINSUR+CLMSEX+SEATBELT',data = claimants).fit()
print(logit_model.summary())

print(logit_model.pred_table())


print (np.exp(logit_model.params))

predict=logit_model.predict(pd.DataFrame(claimants[['CLMAGE','LOSS','CLMINSUR','CLMSEX','SEATBELT']]))
from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(claimants['ATTORNEY'],predict > 0.5 )
print(cnf_matrix)

accuracy = (435+506)/(435+250+149+506)
print(accuracy)

#changing feature

logit_model1 = sm.logit('ATTORNEY~CLMAGE+LOSS+CLMINSUR+CLMSEX',data = claimants).fit()
print(logit_model1.summary())

print(logit_model1.pred_table())

predict1=logit_model1.predict(pd.DataFrame(claimants[['CLMAGE','LOSS','CLMINSUR','CLMSEX']]))
from sklearn.metrics import confusion_matrix
cnf_matrix1 = confusion_matrix(claimants['ATTORNEY'],predict > 0.5 )
print(cnf_matrix1)

accuracy1 = (430+510)/(430+255+145+510)
print(accuracy1)

#removing another feature

logit_model1 = sm.logit('ATTORNEY~LOSS+CLMINSUR+CLMSEX',data = claimants).fit()
print(logit_model1.summary())

print(logit_model1.pred_table())

predict1=logit_model1.predict(pd.DataFrame(claimants[['LOSS','CLMINSUR','CLMSEX']]))
from sklearn.metrics import confusion_matrix
cnf_matrix1 = confusion_matrix(claimants['ATTORNEY'],predict > 0.5 )
print(cnf_matrix1)

accuracy=(429+508)/(429+256+149+506)
print(accuracy)




