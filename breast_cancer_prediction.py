from sklearn.datasets import load_breast_cancer 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

breast_cancer_data = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size=0.2, random_state=100)

best_score=0
for k in range(1,101):
    print('K: ', k)   
    classifier = KNeighborsClassifier(n_neighbors= k)
    classifier.fit(x_train, y_train)
    score = classifier.score(x_test, y_test)
    if score>best_score:
        best_score = score
        best_k = k
    print(score)

print(best_k)