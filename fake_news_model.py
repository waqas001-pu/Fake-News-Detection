import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

true_data=pd.read_csv('/kaggle/input/fake-news-detection-datasets/News _dataset/True.csv')
fake_data=pd.read_csv('/kaggle/input/fake-news-detection-datasets/News _dataset/Fake.csv')

print(true_data[0:10])

fake_data[0:4]
print(true_data['text'][0])

true_data.columns
len(true_data)
len(fake_data)
fake_data['label'] = 1
true_data['label']=0
print(true_data.head())

fake_data.head()
all_data=pd.concat([fake_data, true_data])
random_permutation = np.random.permutation(len(all_data))
all_data= all_data.iloc[random_permutation]
print(all_data.columns)
all_data.head()
filterd_data=all_data.loc[:, ['title', 'text', "subject", 'label']]
filterd_data.head()
filterd_data.isnull().sum()
filterd_data['training_feature']=filterd_data['title']+' '+filterd_data['text']+' '+filterd_data['subject']
filterd_data.head()
X= filterd_data['training_feature'].values
y = filterd_data['label']
l_X=filterd_data['training_feature'].values[0:1000]
l_Y= filterd_data['label'].values[0:1000]
print(l_X.shape)
print(l_Y.shape)
type (l_X)
print(X[0:1])
# from sklearn import preprocessing

# print(X.shape)
# print(type(X))
# # print(X[0:1])
# X=X.reshape((1,-1))
# X = preprocessing.normalize(X)
# np.random.shuffle(X)
# print(X.shape)
# print(type(X))

# X = preprocessing.normalize([X.reshape(1,-1)])
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
vectorizer= TfidfVectorizer()
X=vectorizer.fit_transform(X)
l_vectorizer= TfidfVectorizer()
l_X=l_vectorizer.fit_transform(l_X)
print(type(X))
print(X.shape)
print(type(l_X))
print(l_X.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state=42)
l_X_train, l_X_test, l_Y_train, l_Y_test = train_test_split(l_X, l_Y, test_size = 0.2, random_state=42)
X_train.shape
model=LogisticRegression()
model.fit(X_train,Y_train)
# Model accuracy on the test set
test_y_hat=model.predict(X_test)
print(accuracy_score(test_y_hat, Y_test))

# Model Accuracy on  training set
train_y_hat = model.predict(X_train)
print(accuracy_score(train_y_hat,Y_train))
def predict(input_data):
    
    y_hat= model.predict(input_data)
    if y_hat==0:
        return "The article is Fake"
    else:
        return 'The article is not Fake'

print(predict(X_test[2000]))
model.predict(X_test[2000])[0]
from sklearn.metrics import precision_score, recall_score, f1_score

print (precision_score(Y_train, train_y_hat,))
print (recall_score(Y_train, train_y_hat,))
print(f1_score(Y_train, train_y_hat,))


print (precision_score(Y_test, test_y_hat,))
print (recall_score(Y_test, test_y_hat,))
print(f1_score(Y_test, test_y_hat,))

from sklearn.metrics import confusion_matrix
print ([['TP', 'FP'],['FN', 'TN']])
print(confusion_matrix(Y_test, test_y_hat,))
print (confusion_matrix(Y_train, train_y_hat,))
from sklearn import svm

#Create a svm Classifier
model = svm.SVC(kernel='linear') # Linear Kernel
clf_poly=clf = svm.SVC(kernel='poly') # Polynomial kernel
print(X_train.shape)
# X_train.iloc[0:1000].shape
#Train the model using the training sets
model.fit(l_X_train, l_Y_train)
# clf_poly.fit(X_train,Y_train)
#Predict the response for test dataset
y_pred = model.predict(l_X_test)
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

print(accuracy_score(y_pred, l_Y_test))
print (precision_score(l_Y_test, y_pred,))
print (recall_score(l_Y_test, y_pred,))
print(f1_score(l_Y_test, y_pred,))
print(confusion_matrix(l_Y_test, y_pred,))
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(l_X_train, l_Y_train)
y_pred = model.predict(l_X_test)
print (y_pred)
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score,confusion_matrix


print(accuracy_score(y_pred, l_Y_test))
print (precision_score(l_Y_test, y_pred,))
print (recall_score(l_Y_test, y_pred,))
print(f1_score(l_Y_test, y_pred,))
print(confusion_matrix(l_Y_test, y_pred,))
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
# X_train.todense()
# model2=GaussianNB()
# model2.fit(X_train.toarray(), Y_train)
y_pred

model.fit(l_X_train.toarray(), l_Y_train)
y_pred = model.predict(l_X_test.toarray())
print(accuracy_score(y_pred, l_Y_test))
print (precision_score(l_Y_test, y_pred,))
print (recall_score(l_Y_test, y_pred,))
print(f1_score(l_Y_test, y_pred,))
print(confusion_matrix(l_Y_test, y_pred,))
