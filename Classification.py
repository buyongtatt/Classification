import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
import graphviz
from IPython.display import Image
import pydotplus
from sklearn.tree import export_graphviz

# IMPORT EXTERNAL DATA FILE
dataset = pd.read_csv("Students' Academic Performance.csv")

print(dataset)

# CONVERT NON-NUMERICAL DATA INTO NUMERICAL DATA
category_data = ["gender", 'NationalITy', 'PlaceofBirth', 'StageID', 'GradeID', 'SectionID', 'Topic',
                 'Semester', 'Relation', 'ParentAnsweringSurvey', 'ParentschoolSatisfaction', 'StudentAbsenceDays', 'Class']

encoder = LabelEncoder()
for i in category_data:
    dataset[i] = encoder.fit_transform(dataset[i])
    dataset.dtypes

print(dataset.dtypes)

# split data into training dataset and testing dataset

# determine independent variable(x) and dependent variable(y)
independent_variable = ['gender', 'NationalITy', 'PlaceofBirth', 'StageID', 'GradeID', 'SectionID', 'Topic',
                        'Semester', 'Relation', 'ParentAnsweringSurvey', 'ParentschoolSatisfaction', 'StudentAbsenceDays', ]

dependent_variable = ['Class']

x = dataset[independent_variable]
y = dataset[dependent_variable]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)
print("This is training dataset model (80%):")
print(x_train)
print("This is testing dataset model (20%):")
print(x_test)
