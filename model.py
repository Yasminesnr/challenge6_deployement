# Import the libraries
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
np.random.seed(0)

import pickle

#Loading the dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['Species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

#Creating training and testing data (this is a way of splitting the dataset)
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75

# Creating dataframes with training rows and testing rows
train, test = df[df['is_train'] == True], df[df['is_train'] == False]

# Create a list of the feature columns names
features = df.columns[:4]

# Converting each species name into digits
y = pd.factorize(train['Species'])[0]

# creating a random forest classifier
clf = RandomForestClassifier(n_jobs = 2, random_state = 0)

# training the model
clf.fit(train[features], y)

#Applying the trained Classifier in the test
y_pred = clf.predict(test[features])

# mapping names for the plants for each predictes plant class
preds = iris.target_names[y_pred]

#evaluate the model
print(accuracy_score(test['Species'] ,preds))

# Make pickle file of our model
pickle.dump(clf, open("model.pkl","wb"))