import os

import numpy as np
import pandas as pd
from plotnine import ggplot, ggsave, aes, geom_line, geom_point, geom_histogram
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Loading data
path = "./data/car.data"
names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
dataset = pd.read_csv(path, names=names)

# Printing information
print('\nFirst 5 records: ')
print(dataset.head(5))

print('\nSet description: ')
print(dataset.describe())

# Splitting attributes
conditional_attributes = dataset.iloc[:, :6]
decision_attribute = dataset['class']

# Printing information
print('\nConditional attributes description: ')
print(conditional_attributes.describe())

print('\nDecision attribute description: ')
print(decision_attribute.describe())

# Creating directory for files with plots if not exists
if not os.path.isdir('./analyze'):
    os.mkdir('analyze')

# Generating and saving plots for every attributes
for i in conditional_attributes:
    plot = ggplot(dataset, aes(x=i, fill=decision_attribute.name)) + geom_histogram(stat="count")
    filename = '{0}-vs-class.png'.format(i)
    ggsave(plot=plot, filename=filename, dpi=300, scale=1, verbose=False, path='analyze')

for i in conditional_attributes:
    plot = ggplot(dataset, aes(x=decision_attribute.name, fill=i)) + geom_histogram(stat="count")
    filename = 'class-vs-{0}.png'.format(str(i))
    ggsave(plot=plot, filename=filename, dpi=300, scale=1, verbose=False, path='analyze')

# Encoding attributes
encoder = LabelEncoder()
for i in dataset.columns:
    dataset[i] = encoder.fit_transform(dataset[i])

normalize_conditional_attributes = dataset.iloc[:, :6]
normalize_decision_attribute = dataset['class']

# Splitting dataset
X_train, X_test, Y_train, Y_test = \
    train_test_split(normalize_conditional_attributes, normalize_decision_attribute, test_size=0.2, random_state=0)

print("\nTrain set size : " + str(X_train.size))
print("\nTest set size : " + str(X_test.size))

# Calculating results for each parameters
lr_scores = []
knn_scores = []
dtc_scores = []

random_class = np.random.choice(normalize_decision_attribute, len(Y_test))
rnd_score = accuracy_score(Y_test, random_class)

for i in range(1, 11):
    lr = LogisticRegression(C=i)
    lr.fit(X_train, Y_train)
    score_lr = lr.score(X_test, Y_test)
    lr_scores.append((i, score_lr))

for i in range(1, 51):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, Y_train)
    score_knn = knn.score(X_test, Y_test)
    knn_scores.append((i, score_knn))

for i in range(1, 11):
    dtc = DecisionTreeClassifier(max_depth=i)
    dtc.fit(X_train, Y_train)
    score_dtc = dtc.score(X_test, Y_test)
    dtc_scores.append((i, score_dtc))

# Choosing the best results
best_lr = max(lr_scores, key=lambda item: item[1])
best_knn = max(knn_scores, key=lambda item: item[1])
best_dtc = max(dtc_scores, key=lambda item: item[1])

# Printing information about best results
print("\nBest results : ")
print(" - For Random : " + str(rnd_score))
print(" - For Logistic Regression : " + str(best_lr[1]) + " : for parameter C : " + str(best_lr[0]))
print(" - For KNeighbors Classifier : " + str(best_knn[1]) + " : for parameter n_neighbors : " + str(best_knn[0]))
print(" - For Decision Tree Classifier : " + str(best_dtc[1]) + " : for parameter max_depth : " + str(best_dtc[0]))

# Creating directory for files with plots if not exists
if not os.path.isdir('./result'):
    os.mkdir('result')

# Generating pandas datasets from results
columns = ['param', 'score']
dataset_lr = pd.DataFrame.from_records(lr_scores, columns=columns)
dataset_knn = pd.DataFrame.from_records(knn_scores, columns=columns)
dataset_dtc = pd.DataFrame.from_records(dtc_scores, columns=columns)

# Saving plots into files
plot_lr = ggplot(dataset_lr, aes(x=columns[0], y=columns[1])) + geom_line() + geom_point()
plot_knn = ggplot(dataset_knn, aes(x=columns[0], y=columns[1])) + geom_line() + geom_point()
plot_dtc = ggplot(dataset_dtc, aes(x=columns[0], y=columns[1])) + geom_line() + geom_point()
ggsave(plot=plot_lr, filename='lr.png', dpi=300, scale=1, verbose=False, path='result')
ggsave(plot=plot_knn, filename='knn.png', dpi=300, scale=1, verbose=False, path='result')
ggsave(plot=plot_dtc, filename='dtc.png', dpi=300, scale=1, verbose=False, path='result')
