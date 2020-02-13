import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

train_dataset = pd.read_csv("data/train.csv")
test_dataset = pd.read_csv("data/test.csv")
dataset = train_dataset.append(test_dataset, sort=True)

# #### Imputing Age ####

# Extracting titles
dataset['Title'] = dataset['Name']
for names in dataset['Name']:
    dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.', expand=True)

# Replacing titles
mapping = {
    'Mlle': 'Miss',
    'Major': 'Mr',
    'Col': 'Mr',
    'Sir': 'Mr',
    'Don': 'Mr',
    'Mme': 'Miss',
    'Jonkheer': 'Mr',
    'Lady': 'Mrs',
    'Capt': 'Mr',
    'Countess': 'Mrs',
    'Ms': 'Miss',
    'Dona': 'Mrs'
}
dataset.replace({'Title': mapping}, inplace=True)
titles = ['Dr', 'Master', 'Miss', 'Mr', 'Mrs', 'Rev']

# Supplementing age
for title in titles:
    age_to_impute = dataset.groupby('Title')['Age'].median()[titles.index(title)]
    dataset.loc[(dataset['Age'].isnull()) & (dataset['Title'] == title), 'Age'] = age_to_impute

dataset.drop('Title', axis=1, inplace=True)
train_dataset['Age'] = dataset['Age'][:891]
test_dataset['Age'] = dataset['Age'][891:]

# #### Imputing family size #####
dataset['FamilySize'] = dataset['Parch'] + dataset['SibSp']

train_dataset['FamilySize'] = dataset['FamilySize'][:891]
test_dataset['FamilySize'] = dataset['FamilySize'][891:]

# #### Imputing family survival #####
# Based on https://www.kaggle.com/shunjiangxu/blood-is-thicker-than-water-friendship-forever
dataset['Last_Name'] = dataset['Name'].apply(lambda x: str.split(x, ",")[0])
dataset['Fare'].fillna(dataset['Fare'].mean(), inplace=True)

DEFAULT_SURVIVAL_VALUE = 0.5
dataset['Family_Survival'] = DEFAULT_SURVIVAL_VALUE

for grp, grp_df in dataset[['Survived', 'Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',
                            'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):
    if len(grp_df) != 1:
        for ind, row in grp_df.iterrows():
            smax = grp_df.drop(ind)['Survived'].max()
            smin = grp_df.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            if smax == 1.0:
                dataset.loc[dataset['PassengerId'] == passID, 'Family_Survival'] = 1
            elif smin == 0.0:
                dataset.loc[dataset['PassengerId'] == passID, 'Family_Survival'] = 0

for _, grp_df in dataset.groupby('Ticket'):
    if len(grp_df) != 1:
        for ind, row in grp_df.iterrows():
            if (row['Family_Survival'] == 0) | (row['Family_Survival'] == 0.5):
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if smax == 1.0:
                    dataset.loc[dataset['PassengerId'] == passID, 'Family_Survival'] = 1
                elif smin == 0.0:
                    dataset.loc[dataset['PassengerId'] == passID, 'Family_Survival'] = 0

train_dataset['Family_Survival'] = dataset['Family_Survival'][:891]
test_dataset['Family_Survival'] = dataset['Family_Survival'][891:]

# #### Fare quantile-based discretization ####
dataset['Fare'].fillna(dataset['Fare'].median(), inplace=True)
dataset['FareBin'] = pd.qcut(dataset['Fare'], 5)

label = LabelEncoder()
dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])

train_dataset['FareBin_Code'] = dataset['FareBin_Code'][:891]
test_dataset['FareBin_Code'] = dataset['FareBin_Code'][891:]

train_dataset.drop(['Fare'], 1, inplace=True)
test_dataset.drop(['Fare'], 1, inplace=True)

# #### Age quantile-based discretization ####
dataset['AgeBin'] = pd.qcut(dataset['Age'], 4)

label = LabelEncoder()
dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])

train_dataset['AgeBin_Code'] = dataset['AgeBin_Code'][:891]
test_dataset['AgeBin_Code'] = dataset['AgeBin_Code'][891:]

train_dataset.drop(['Age'], 1, inplace=True)
test_dataset.drop(['Age'], 1, inplace=True)

# #### Mapping sex ####
train_dataset['Sex'].replace('male', 0, inplace=True)
train_dataset['Sex'].replace('female', 1, inplace=True)
test_dataset['Sex'].replace('male', 0, inplace=True)
test_dataset['Sex'].replace('female', 1, inplace=True)

# #### Clean datasets ####
train_dataset.drop(['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
test_dataset.drop(['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

# #### Training #####
X = train_dataset.drop('Survived', 1)
y = train_dataset['Survived']
X_Test = test_dataset.copy()

std_scaler = StandardScaler()
X = std_scaler.fit_transform(X)
X_Test = std_scaler.fit_transform(X_Test)

# #### Predict ####
n_neighbors = list(range(1, 50, 1))
algorithm = ['auto']
weights = ['uniform', 'distance']
leaf_size = list(range(1, 50, 1))
hyperparams = {'algorithm': algorithm, 'weights': weights, 'leaf_size': leaf_size, 'n_neighbors': n_neighbors}
gd = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=hyperparams, verbose=True,
                  cv=10, scoring="roc_auc", n_jobs=10)
gd.fit(X, y)

print(gd.best_score_)
print(gd.best_estimator_)

gd.best_estimator_.fit(X, y)
y_pred = gd.best_estimator_.predict(X_Test)

# knn = KNeighborsClassifier(algorithm='auto', leaf_size=26, metric='minkowski',
#                            metric_params=None, n_jobs=1, n_neighbors=6, p=2,
#                            weights='uniform')
# knn.fit(X, y)
# y_pred = knn.predict(X_Test)

result = pd.DataFrame(pd.read_csv("data/test.csv")['PassengerId'])
result['Survived'] = y_pred
result.to_csv("data/submission.csv", index=False)
