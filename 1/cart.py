from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


digits = load_digits()
data = digits.data

train_x, test_x, train_y, test_y = train_test_split(data, digits.target, test_size=0.25, random_state=33)

ss = preprocessing.StandardScaler()
train_ss_x = ss.fit_transform(train_x)
test_ss_x = ss.transform(test_x)

cart = DecisionTreeClassifier()
cart.fit(train_ss_x, train_y)
predict_y = cart.predict(test_ss_x)
print('CART ACC: %0.4lf' % accuracy_score(predict_y, test_y))