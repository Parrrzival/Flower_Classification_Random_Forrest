#Importing Libraries:
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
np.random.seed(0)

#Loading Data:
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target,iris.target_names)

#Splitting Data into training set and test set:
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
train, test = df[df['is_train']==True], df[df['is_train']==False]
print('train = ',len(train),'test = ',len(test))

#converting names of flowers into numbers making it easy for the computer to understand:
features = df.columns[:4]
y = pd.factorize(train['species'])[0]

#training and fitting the model:
clf = RandomForestClassifier(n_jobs=2, random_state=0)
clf.fit(train[features], y)

#Making Predictions
y1 = print("y1 = ",clf.predict(test[features]))
preds = iris.target_names[clf.predict(test[features])]
print("preds = ",preds)

#Confusion Matrix To check Accuracy:
print(pd.crosstab(test['species'],preds, rownames=['Actual Species'],colnames=['Predicted Species']))