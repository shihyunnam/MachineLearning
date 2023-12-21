from sklearn.datasets import load_iris
iris = load_iris()
irisDataInput = iris.data

# feature_names': ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

from sklearn.datasets import load_breast_cancer
bCancer = load_breast_cancer()
# print(bCancer.data)


#splitting train dataset and testing dataset
from sklearn.model_selection import train_test_split
X = iris.data
Y = iris.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=42)
# print(X_train.shape)

#perform knn 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics#성능
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)
scores = metrics.accuracy_score(Y_test,Y_pred)
print(scores)
