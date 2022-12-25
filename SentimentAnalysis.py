#sentiment analysis
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC

df=pd.read_csv('amazon_baby.csv')
df=df.dropna()
np.random.seed(34)
df1=df.sample(frac=0.3)

df1['sentiments']=df.rating.apply(lambda x:0 if x in [1,2] else 1)
x=df1['review']
y=df1['sentiments']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.05,random_state=0)

cv=CountVectorizer()
ctmTr=cv.fit_transform(x_train)
x_test_dtm=cv.transform(x_test)

model=SVC()
model.fit(ctmTr,y_train)
score=model.score(x_test_dtm,y_test)
pred=model.predict(x_test_dtm)
print('Accuracy:',score)
print('Prediction:',pred)
