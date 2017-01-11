import numpy as np
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegressionCV

from sklearn.decomposition import KernelPCA

SEED = 10

"""Load dataset"""
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['num-pregrant', 'plasma-concentration', 'diastolic-bp', 'tricep-skin-thickness', 'serum-insulin', 'bmi', 'pedigree', 'age', 'class']
dataset = pandas.read_csv(url, names=names)

"""Summarize dataset"""
def summarize(dataset):
    print(dataset.shape)
    print(dataset.head(20))
    print(dataset.describe())
    print(dataset.groupby('class').size())
dataset.shape
dataset.head(20)
dataset.describe()
dataset.groupby('class').size()

"""Data Visualization"""
dataset.plot(kind='box', layout=(3,3), figsize=(7, 6), subplots=True, sharex=False, sharey=False)
plt.show()

dataset.hist(figsize=(8, 7))
plt.show()

scatter_matrix(dataset, figsize=(13, 9), diagonal='kde')
plt.show()


dataset[dataset["serum-insulin"]!=0].plot(kind="scatter", x="bmi", y="serum-insulin", c="class", s=10)
dataset[dataset["num-pregrant"]>4].plot(kind="scatter", x="num-pregrant", y="serum-insulin", c="class")
dataset.head(20)

"""Evaluate Algorithms"""
def split(dataset, num_attr=len(names)-1, validation_size=0.20, seed=SEED):
    array = dataset.values
    X = array[:,:num_attr]
    Y = array[:,num_attr]
    X_train, X_validation, Y_train, Y_validation = cross_validation.train_test_split(X, Y, test_size=validation_size, random_state=seed)
    return X_train, X_validation, Y_train, Y_validation

def evaluate(X_train, Y_train, models, num_folds=10, scoring='accuracy', seed=SEED):
    num_instances = len(X_train)
    results = []
    names = []
    for name, model in models:
        kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
        cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg ="%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
    return names, results

def compare(names, results, figsize=None):
    fig = plt.figure(figsize=figsize)
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()

X_train, X_validation, Y_train, Y_validation = split(dataset)

"""Preprocess"""
means = X_train.mean(axis=0)
X_train -= means
X_validation -= means

# """PCA"""
# pca = KernelPCA()
# X_train = pca.fit_transform(X_train)
# X_validation = pca.transform(X_validation)

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('QDA', QuadraticDiscriminantAnalysis()))
models.append(('ABC', AdaBoostClassifier()))
models.append(('MLP', MLPClassifier()))
models.append(('SVM RBF', SVC(gamma=2, C=1)))
# models.append(('GPC', GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)))
# models.append(('SGD', SGDClassifier(loss='log')))
models.append(('LRCV', LogisticRegressionCV()))

names, results = evaluate(X_train, Y_train, models)
compare(names, results, figsize=(8, 4))

"""Make Predictions"""
def score(labels, predictions):
    print(accuracy_score(labels, predictions))
    print(confusion_matrix(labels, predictions))
    print(classification_report(labels, predictions))

model = LogisticRegression()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
score(Y_validation, predictions)

model = LinearDiscriminantAnalysis()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
score(Y_validation, predictions)

model = LogisticRegressionCV()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
score(Y_validation, predictions)


"""Tune Algorithms"""
model = LogisticRegression()
model.fit(X_train, Y_train)
pred_train = model.predict(X_train)
pred_validation = model.predict(X_validation)
accuracy_score(Y_train, pred_train)
accuracy_score(Y_validation, pred_validation)


model = RandomForestClassifier(random_state=SEED, n_estimators=10, min_samples_split=2, min_samples_leaf=1)
kfold = cross_validation.KFold(X_train.shape[0], n_folds=3, random_state=SEED)
scores = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold)
scores.mean()

model = RandomForestClassifier(random_state=SEED, n_estimators=50, min_samples_split=4, min_samples_leaf=2)
kfold = cross_validation.KFold(X_train.shape[0], n_folds=3, random_state=SEED)
scores = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold)
scores.mean()

model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
accuracy_score(Y_validation, predictions)

#Consider using PCA to find proper components
#Don't forget to demean
#Also look into how to encode categorical variables
#Set aside a test set and don't touch until final check
#Use different validation set sizes to determine bias/variance
#Finally try out XGBoost


#
#
# def learning_curve(model): # Flawed in that can have such a small validation set that it will always be correct
#     validation_sizes = np.arange(0.05, 0.4, 0.05)
#     training_accuracy = []
#     validation_accuracy = []
#     for size in validation_sizes:
#         X_train, X_validation, Y_train, Y_validation = split(dataset, validation_size=size)
#         model.fit(X_train, Y_train)
#         pred_train = model.predict(X_train)
#         pred_validation = model.predict(X_validation)
#         training_accuracy.append(accuracy_score(Y_train, pred_train))
#         validation_accuracy.append(accuracy_score(Y_validation, pred_validation))
#
# training_accuracy
#
# validation_accuracy
#
#
#
# """
# http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
# models = [
#     KNeighborsClassifier(3),
#     SVC(kernel="linear", C=0.025),
#     SVC(gamma=2, C=1),
#     GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
#     DecisionTreeClassifier(max_depth=5),
#     RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
#     MLPClassifier(alpha=1),
#     AdaBoostClassifier(),
#     GaussianNB(),
#     QuadraticDiscriminantAnalysis()]
# """
