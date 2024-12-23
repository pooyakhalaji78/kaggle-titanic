import pandas as pd #to read csv data
import numpy as np #for math
import seaborn as sns #for visualization
import matplotlib.pyplot as plt #for visualization
from sklearn.model_selection import train_test_split #for splitting data

from sklearn.linear_model import LogisticRegression #Machine Learning model (logistic regression)
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.ensemble import GradientBoostingClassifier #Gradient Boosting
from xgboost import XGBClassifier #XGBoost
from sklearn.svm import SVC #Support Vector Machine
from sklearn.neighbors import KNeighborsClassifier #K-Nearest Neighbors
from sklearn.naive_bayes import GaussianNB #Naive Bayes
from sklearn.neural_network import MLPClassifier #Neural Network

from sklearn.ensemble import StackingClassifier #Stacking
from sklearn.svm import SVC #Stacking

from sklearn.metrics import accuracy_score, confusion_matrix #for evaluating the model

'''
Train.csv:
891 passengers
1:survived 0:dead
Pclass: 1(upper class), 2(middle class), 3(lower class)

sibsp: number of siblings aboard the titanic 
sibling: [brother, sister, stepbrother, stepsister]
spouse: [husband, wife]

parch: number of parents/children aboard the titanic
parent: [mother, father]
child: [son, daughter, stepson, stepdaughter]
Some children travelled only with a nanny, therefore parch=0 for them.

embarked: C=Cherbourg, Q=Queenstown, S=Southhampton

Test.csv:
should be done after the training is done.

gender_submission.csv:
as an example of how to upload the predictions
'''

#-------------------read data-----------------------
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

##properties for analyzing data
#print(train_data.head()) #shows the first 5 rows
#print(train_data.tail()) #shows the last 5 rows
#print(train_data.info()) #shows the info of the data
#print(train_data.describe()) #shows the summary of the data

##shows columns with missing data
#sns.heatmap(train_data.isnull(), cbar=False)
#plt.show()

#-----------------Data Preprocessing----------------
#filling missing "Age" with median
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
#filling the test also with train data cause it is safer
test_data['Age'] = test_data['Age'].fillna(train_data['Age'].median())

#filling missing "Embarked" with mode 
train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])
test_data['Embarked'] = test_data['Embarked'].fillna(train_data['Embarked'].mode()[0])

# Filling missing "Fare" in test data with the median from the training data
test_data['Fare'] = test_data['Fare'].fillna(train_data['Fare'].median())

#dropping "Cabin" column because too many missing data
train_data.drop('Cabin', axis=1, inplace=True) 
test_data.drop('Cabin', axis=1, inplace=True)

#Converting categorical data to numerical data (One Hot Encoding)
train_data = pd.get_dummies(train_data, columns=['Sex', 'Embarked'], drop_first=True)
test_data = pd.get_dummies(test_data, columns=['Sex', 'Embarked'], drop_first=True)

# Ensure both DataFrames have the same columns after one-hot encoding
test_data = test_data.reindex(columns=train_data.columns.drop('Survived'), fill_value=0)

#dropping irrelevant columns
train_data.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)
test_data.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

##shows columns with missing data
# sns.heatmap(train_data.isnull(), cbar=False)
# sns.heatmap(test_data.isnull(), cbar=False)
# plt.show()

#-----------------Split Data for Training and Validation----------------
#A Table without "Survived" column
x = train_data.drop('Survived', axis=1)
# Table only with "Survived" column
y = train_data['Survived']
#test_size = 20% validation
#random_state = saves the random number
#if you want it to be random everytime, set it to None
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

#-----------------Train a ML Model----------------
#Logistic Regression - 81%
# model = LogisticRegression(max_iter = 1000)

#Decision Tree - 79%
# model = DecisionTreeClassifier(max_depth=5, random_state=42)

#Random Forest - 82% (Ensemble of Decision Trees)
# model = RandomForestClassifier(n_estimators=1000, random_state=42)

#Gradient Boosting - 80%
# model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

#XGBoost - 82%
# model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

#SVM - 65%
# model = SVC(kernel='rbf', C=1.0, random_state=42)

#K-Nearest Neighbors (KNN) - 71%
# model = KNeighborsClassifier(n_neighbors=5)

#Naive Bayes - 77%
# model = GaussianNB()

#Neural Network - 76%
# model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)



#-----------------Stacking (To train all models)----------------
base_learners = [
    ('lr', LogisticRegression(max_iter=1000)),
    ('dt', DecisionTreeClassifier(max_depth=5, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=1000, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)),
    ('xgb', XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)),
    ('svc', SVC(kernel='rbf', C=1.0, probability=True, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('nb', GaussianNB()),
    ('nn', MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42))
]

# Define the meta-model (could be Logistic Regression or another model)
#Logistic Regression - 81%
#Decision Tree - 73%
#Random Forest - 82%
#Gradient Boosting - 78%
#XGBoost - 78%
#SVM - 80%
#K-Nearest Neighbors (KNN) - 81%
#Naive Bayes - 78%
#Neural Network - 82%
meta_model = MLPClassifier()

# Create the stacking classifier
stacking_model = StackingClassifier(estimators=base_learners, final_estimator=meta_model)





#-----------------Evaluating the model----------------

#Normal
#model.fit(x_train, y_train)
#y_pred = model.predict(x_val)

#Stacking
stacking_model.fit(x_train, y_train)
y_pred = stacking_model.predict(x_val)
#-----------------Preparing Test Set----------------
#Normal
# predictions = model.predict(test_data)

#Stacking
predictions = stacking_model.predict(test_data)

#-----------------Printing Results----------------
print('Accuracy:', accuracy_score(y_val, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_val, y_pred))
#-----------------Submit the results----------------
submission = pd.DataFrame({
    'PassengerId': pd.read_csv('test.csv')['PassengerId'],
    'Survived': predictions
})
submission.to_csv('submission.csv', index=False)

