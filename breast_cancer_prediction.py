from sklearn.datasets import load_breast_cancer 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score


#loading the datasete
breast_cancer_data = load_breast_cancer()

#spliting the dataset
x_train, x_test, y_train, y_test = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size=0.2, random_state=100)

#training with the split data
classifier = KNeighborsClassifier(n_neighbors= 23)
classifier.fit(x_train, y_train)

#making an prediction
y_predict = classifier.predict(x_test)

#accuracy score
print(100*accuracy_score(y_test, y_predict),'%')

#calculating the F1 score
precision = precision_score(y_test, y_predict)
recall= recall_score(y_test, y_predict)
print('F1 score: ', 2*((precision*recall)/(precision+recall)))