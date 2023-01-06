import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV , train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix , classification_report, accuracy_score

train_file = r'S:\\Programming languages\\Machine Learning\\01_Tweet_Spam_Ham\\train_tweet.csv'
test_file = r'S:\\Programming languages\\Machine Learning\\01_Tweet_Spam_Ham\\test_tweet.csv'

df = pd.read_csv(train_file)
wf = pd.read_csv(test_file)

def vectorizer(data):
    vect = TfidfVectorizer(ngram_range=(1,1))
    x = vect.fit_transform(data)
    x = vect.get_feature_names_out()
    return x


for column in df.columns:
    df[column] = df[column].replace(r'\W'," ",regex=True)
    df[column] = df[column].replace(' ','',regex=True)

x = pd.DataFrame(df['tweet'])
y = pd.DataFrame(df['tweet'])
x_test = pd.DataFrame(wf['tweet'])

x_train , x_test , y_train , y_test = train_test_split(x,y,random_state=23,test_size=50)

params = {'C':[1,2,3,5,6,10,15],'max_iter':[50,100]}

# x_train = vectorizer(x_train)
# y_train = vectorizer(y_train)

lg = LogisticRegression(C=10,max_iter=100)
lg_model = GridSearchCV(lg,params,scoring='f1',cv=5,error_score='raise')

lg_model.fit(x_train,y_train)

print(lg_model.best_score_)
print(lg_model.best_params_)