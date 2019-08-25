import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib


###################################Creating Dataset and including the first row by setting no header as input
dataset = pd.read_csv('eegf.csv', header = None)

###################################Renaming the columns
dataset.columns = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
#print('Shape of the dataset: ' + str(dataset.shape))
#print(dataset.head())

#Creating the dependent variable class
factor = pd.factorize(dataset['L'])
dataset.L = factor[0]
definitions = factor[1]
#print(dataset.L.head())
#print(definitions)

################################Splitting the data into independent and dependent variables
X = dataset.iloc[:,0:11].values
y = dataset.iloc[:,11].values
#print('The independent features set: ')
#print(X[:11,:])
#print('The dependent variable: ')
#print(y[:11])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


classifier = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

reversefactor = dict(zip(range(10),definitions))
y_test = np.vectorize(reversefactor.get)(y_test)
y_pred = np.vectorize(reversefactor.get)(y_pred)

# Making the Confusion Matrix
for i in range(len(y_test)):
    print(y_test[i],":::::", y_pred[i])
print(pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted']))
print(accuracy_score(y_test, y_pred))
