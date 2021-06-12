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
