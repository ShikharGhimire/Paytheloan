import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#Importing the libraries
dataset = pd.read_csv('loan_data.csv')

#Getting out the info from the dataset
dataset.info()

#Description of the dataset
dataset.describe()
#Observing the header in the dataset
dataset.head()

#Exploring the data for credit policy and fico
plt.figure(figsize = (10,6))
dataset[dataset['credit.policy']==1]['fico'].hist(alpha=0.5, color = 'blue',bins = 30,label = 'credit.Policy=1')
dataset[dataset['credit.policy']==0]['fico'].hist(alpha=0.5,color = 'red',bins = 30,label = 'credit.Policy = 0')
plt.legend()
plt.xlabel('FICO')

#Exploring the data for fully paid and not fully paid 
plt.figure(figsize = (10,6))
dataset[dataset['not.fully.paid']==1]['fico'].hist(alpha = 0.5, color = 'blue',bins = 30,label = 'not.fully.paid=1')
dataset[dataset['not.fully.paid']==0]['fico'].hist(alpha = 0.5, color = 'red',bins = 30, label = 'not.fully.paid=0')
plt.legend()
plt.xlabel('FICO')

#Creating a counterplot using seaborn to see the count of loans given by the purpose with the colour hue defined by  
#not fully paid
plt.figure(figsize=(11,7))
sns.countplot(x='purpose',hue = 'not.fully.paid',data=dataset,palette = 'Set1')
#We can see that the loan has not been fully paid for almost all purpose

#Converting the categorical value using dummy variables and making it the final dataset that we will be working on
categorical = ['purpose']
final_dataset = pd.get_dummies(dataset,columns = categorical,drop_first = True)

#Avoiding  the dummy variable trap by removing the last coloumn of the dummy variable
final_dataset.drop(['purpose_home_improvement'],axis=1)

#Checking out all the info from the final dataset
final_dataset.info() #categorical values were successfully converted to  integer values and there was no null objects present

#Splitting the data into dependent and independent variable
X = final_dataset.drop('not.fully.paid',axis=1)
y = final_dataset['not.fully.paid']

#Splitting the dataset into training and testing set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state=0)

#Checking whether the target value is balanced or imbalanced by joining the training dependent and independent value
#Joining the independent dataset and dependent dataset
joined_trained = pd.concat([X_train,y_train],axis = 1)

#Seperating two sets of class values from dependent and independent value
not_fully_paid = joined_trained[joined_trained['not.fully.paid']==0]
fully_paid = joined_trained[joined_trained['not.fully.paid']==1]

#Counting the number of fully paid and not fully paid values in training set
#6031 not fully paid
 #1152 fully paid
joined_trained['not.fully.paid'].value_counts()

#Uneven number of classes so using oversampling method to even out the classes
from sklearn.utils import resample
oversample_minority = resample(fully_paid,
                               replace = True,
                               n_samples = len(not_fully_paid), #matching it with not fully paid
                               random_state = 0)

#joining the two seperated set of classes together
final_training_data = pd.concat([not_fully_paid,fully_paid])

#Seperating the dependent and independent variable from the final training data set
X_train = final_training_data.drop('not.fully.paid',axis = 1)
y_train = final_training_data['not.fully.paid']
#Feature scaling to make the computation faster
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#Using decision tree algorithm
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier(criterion = 'gini',max_depth = 1,random_state = 0) #Changed from entropy to gini and max_depth was set to 1 after gridsearch CV
decision_tree.fit(X_train,y_train)
#Predicition
decision_tree_predicition = decision_tree.predict(X_test)

#Confusion matrix for the decision tree
from sklearn.metrics import confusion_matrix
cm_decision_tree = confusion_matrix(y_test,decision_tree_predicition)

#Using random forest algorithm in the dataset
from sklearn.ensemble import RandomForestClassifier
randomforest_classifier = RandomForestClassifier(n_estimators = 250,random_state = 0) #changed n_estimators from 19  to 250
randomforest_classifier.fit(X_train,y_train)

#Predicition
random_forest_prediciton = randomforest_classifier.predict(X_test)

#Confusion matrix for the randomforest classifier predicition
from sklearn.metrics import confusion_matrix
cm_randomforest = confusion_matrix(y_test,random_forest_prediciton)

#Using gridsearch CV on decision tree to find out the best hyperparameters and parameters
from sklearn.model_selection import GridSearchCV
tree_parameters = {'criterion':['gini','entropy'],'max_depth':np.arange(1,1000)}
grid_search = GridSearchCV(estimator = decision_tree,
                           param_grid = tree_parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train,y_train)
best_decision_parameters = grid_search.best_params_
best_decision_accuracy = grid_search.best_score_


#Using gridsearch CV on random forest tree to find out the best hyperparameters and parameters for it
#GridSearchCV already imported
parameters = [{'n_estimators':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,50,60,70,80,90,100,200,230,240,250,260]}]
grid_searchtwo = GridSearchCV(estimator=randomforest_classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_searchtwo = grid_searchtwo.fit(X_train,y_train)
best_randomforest = grid_searchtwo.best_params_
best_random_accuracy = grid_searchtwo.best_score_
