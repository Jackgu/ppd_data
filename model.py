#coding=utf-8
import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import model_selection
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,roc_auc_score,roc_curve,auc
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import linear_model, datasets
from sklearn.cross_validation import train_test_split
import load_data


print("===================")
d = load_data.LoadData()

### Generate Y
y= d.loc[:,"IsDefault"].values
count_default = sum(y)
print "bad sample: {0}".format(count_default)
print "Default Rate: {0}%".format(100.0*count_default/len(y))
print("===================")

### Generate X
x_data = d.loc[:,[u"借款金额",u"借款期限", u"借款利率",  u"年龄", u"历史成功借款次数", u"手机认证",	u"户口认证", u"视频认证",	u"学历认证",	u"征信认证",	u"淘宝认证"]]
#, u"初始评级"
x_original= x_data.values

# Prepare Data For Machine Learning in Scikit-Learn
scaler = MinMaxScaler(feature_range=(0, 1))
x = scaler.fit_transform(x_original)
#print(x[0:5])

# Feature Selection For Machine Learning
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(x, y)

np.set_printoptions(precision=3)
#print(fit.scores_)
features = fit.transform(x)
#print(features[0:5,:])
x = features
#print("===================")

### Generate train data
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=20)
print("Testing on  {}".format(len(x_train)))


##### Begin Sklearn######
print "This is a Sklearn Machine Learning Process"

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'  ##http://scikit-learn.org/stable/modules/model_evaluation.html

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
#models.append(('SVM', SVC()))  # a little slow, can add later
results = []
names = []
print "Algorithms: mean (std)"
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

'''
# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
'''
# Make predictions on validation dataset
print("===================")
print "Train & Validate on LogisticRegression"
lr = linear_model.LogisticRegression()
lr.fit(x_train, y_train)
predictions = lr.predict(x_test)

print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions, target_names=["not","default"]))

print("roc_auc_score: {}".format(roc_auc_score(y_test, predictions)))
