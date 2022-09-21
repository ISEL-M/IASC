from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

A = [0x01,0x00,0x00,0x01,0x00,0x01,0x01,0x00,0x00,0x01,0x01,0x00,0x01,0x00,0x00,0x01]
B = [0x01,0x01,0x01,0x01,0x01,0x00,0x00,0x01,0x01,0x00,0x00,0x01,0x01,0x01,0x01,0x01]
X = [A, B]
y = [1, 1]

#X, y = make_classification(n_samples=100,random_state=1)

print(len(X),len(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=(True))

clf = MLPClassifier(
    hidden_layer_sizes=[3],
    momentum=0,    
    learning_rate_init=0.05,
    activation="tanh",
    max_iter=100000,
    tol=1e-10
    ).fit(X_train, y_train)
print(clf.n_iter_)

ok={}

