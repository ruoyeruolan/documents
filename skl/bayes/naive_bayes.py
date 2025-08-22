# -*- encoding: utf-8 -*-
# @Introduce  : 
# @File       : naive_bayes.py
# @Author     : ryrl
# @Email      : ryrl970311@gmail.com
# @Time       : 2025/08/20 23:02
# @Description: 

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0)

clf = GaussianNB()
pred = clf.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != pred).sum()))