import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

dataset=pd.read_csv('C:/Users/Kumar Ambuj/PycharmProjects/DECISIONMINDS/venv/bank_data.csv')
print(dataset.head(5))

#checking  if there is any null data or not
print(dataset.isnull().sum())

x = dataset.drop(["y"],axis=1)
y = dataset["y"]

#x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.3,random_state=1)

#from sklearn.linear_model import LogisticRegression

#logmodel = LogisticRegression()

#print(logmodel.fit(x_train,y_train))

#predictions = logmodel.predict(x_test)

#print(predictions)

#print(predictions.summary())


import numpy as np
import statsmodels.formula.api as sm
logit_model = sm.logit('y~balance+housing+loan+duration+campaign+poutfailure+poutother+poutsuccess+poutunknown+con_cellular+con_telephone+con_unknown+divorced+married+single+johousemaid+jomanagement+joretired+jostudent+jounknown',data = dataset).fit()

print(logit_model.summary())

print(logit_model.pred_table())


print (np.exp(logit_model.params))

predict=logit_model.predict(pd.DataFrame(dataset[['balance','housing','loan','duration','campaign','poutfailure','poutother','poutsuccess','poutunknown','con_cellular','con_telephone','con_unknown','divorced','married','single','johousemaid','jomanagement','joretired','jostudent','jounknown']]))
from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(dataset['y'],predict > 0.5 )
print(cnf_matrix)

accuracy = (39014+1705)/(39014+908+3584+1705)
print(accuracy)




