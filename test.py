logit_model = sm.logit('y~age+default+balance+housing+loan+duration+campaign+pdays+previous+poutfailure+poutother+poutsuccess+poutunknown+con_cellular+con_telephone+con_unknown+divorced+married+single+joadmin+joblue.collar+joentrepreneur+johousemaid+jomanagement+joretired+joself.employed+joservices+jostudent+jotechnician+jounemployed+jounknown',data = dataset).fit()
print(logit_model.summary())