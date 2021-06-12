import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image
import pydotplus
from sklearn.tree import export_graphviz

# IMPORT EXTERNAL DATA FILE
dataset = pd.read_csv("Students' Academic Performance.csv")

print(dataset)

# CONVERT CATEGORICAL DATA INTO NUMERICAL DATA
category_data = ["gender", 'NationalITy', 'PlaceofBirth', 'StageID', 'GradeID', 'SectionID', 'Topic',
                 'Semester', 'Relation', 'ParentAnsweringSurvey', 'ParentschoolSatisfaction', 'StudentAbsenceDays']

encoder = LabelEncoder()
for i in category_data:
    dataset[i] = encoder.fit_transform(dataset[i])
    dataset.dtypes

print(dataset.dtypes)

# split data into training dataset and testing dataset

# determine independent variable(x) and dependent variable(y)
independent_variable = ['gender', 'NationalITy', 'PlaceofBirth', 'StageID', 'GradeID', 'SectionID', 'Topic',
                        'Semester', 'Relation', 'ParentAnsweringSurvey', 'ParentschoolSatisfaction', 'StudentAbsenceDays']

dependent_variable = ['Class']

x = dataset[independent_variable]
y = dataset[dependent_variable]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)
print("This is training dataset model (80%):")
print(x_train)
print("This is testing dataset model (20%):")
print(x_test)

# DO LOGISTIC REGRESSION MODEL
logistic_regression = LogisticRegression(solver="lbfgs",
                                         multi_class="auto", max_iter=5000)
logistic_regression.fit(x_train, y_train)

# SHOW TRAINING DATASET
y_trainPrediction = logistic_regression.predict(x_train)
print(x_train)
print("Prediction list for training dataset: ")
print(y_trainPrediction)

# SHOW ACCURACY OF PREDICTION FOR TRAINING DATASET
print("Accuracy of the prediction for the training dataset: ",
      metrics.accuracy_score(y_train, y_trainPrediction))

# SHOW TESTING DATASET
y_testPrediction = logistic_regression.predict(x_test)
print(x_test)
print("Prediction list for testing dataset: ")
print(y_testPrediction)

# SHOW ACCURACY OF PREDICTION FOR TESTING DATASET
print("Accuracy of the prediction for the testing dataset: ",
      metrics.accuracy_score(y_test, y_testPrediction))

# SHOW TRAINING DATASET DECISION TREE
tree_classifier = DecisionTreeClassifier(criterion="entropy")
tree_classifier.fit(x_train, y_train)
dot_data = export_graphviz(tree_classifier, out_file=None,
                           filled=True, rounded=False,
                           special_characters=True, feature_names=independent_variable)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_jpg('training.jpg')
Image(graph.create_jpg())

# SHOW TESTING DATASET DECISION TREE
tree_classifier = DecisionTreeClassifier(criterion="entropy")
tree_classifier.fit(x_test, y_test)
dot_data1 = export_graphviz(tree_classifier, out_file=None,
                            filled=True, rounded=False,
                            special_characters=True, feature_names=independent_variable)
graph = pydotplus.graph_from_dot_data(dot_data1)
graph.write_jpg('testing.jpg')
Image(graph.create_jpg())
