import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from matplotlib.ticker import ScalarFormatter
from sklearn.metrics import f1_score
 
#%% Function to import the dataset
def import_dataset(file_path):
    dataset = pd.read_csv(file_path, skiprows=(1), header=None)
    return dataset

#%% Function to split the dataset into X and y
def splitData(dataset):
    le = preprocessing.LabelEncoder()
    X=dataset.values[:,:-1] 
    y=dataset.values[:,-1]
    y= le.fit_transform(y)
    return X,y

#%% Function to split the data for training and testing
def trainSplit(X,y):
    X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.3, random_state=55)
    return X_train, X_test, y_train, y_test
   
#%% Function to train the decision tree classifier
def train_classifier(X_train, y_train):
    clf = DecisionTreeClassifier(criterion="gini",max_depth=5, min_samples_leaf=5)
    clf.fit(X_train, y_train)
    return clf

#%% Function to calculate the F1 Score
def calculateFmeasure(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    fmeasure = f1_score(y_test, y_pred, average="weighted")
    return fmeasure

#%% Function to create the bar graph
def create_bar_graph(data_labels, data_before, data_after):
    plt.figure(figsize=(10, 6))
    plt.bar(data_labels, data_after, color='blue')
    plt.xlabel('Scaling Techniques')
    plt.ylabel('Feature Values')
    plt.title('Feature Values Before and After Scaling')
    plt.yscale("log")
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    plt.show()

#%% Import the dataset
path= r #{your_file_path} --- please add your data file path here
dataset= import_dataset(path)

#%% Split the dataset into X and y
X,y= splitData(dataset)

#%% split the data for training and testing (before normalization)
X_train, X_test, y_train, y_test= trainSplit(X, y)

#%% Create bar graph data
data_labels = ['Before', 'Softmax Scaling', 'Max Absolute Scaling', 'Robust Scaling']
data_before = X[:,:].mean()
data_after = []

#%% Train the classifier before normalization
clf_before = train_classifier(X_train, y_train)

#%% Calculate accuracy before normalization
accuracy_before = calculateFmeasure(clf_before, X_test, y_test)

#%% Add accuracy before normalization to the bar graph data
data_after.append(accuracy_before)

#%% Apply softmax scaling
softmax_scaler = MaxAbsScaler()
X_softmax = softmax_scaler.fit_transform(X)

#%% split the data for training and testing (for softmax scaling)
X_train, X_test, y_train, y_test= trainSplit(X_softmax, y)

#%% Train the classifier after softmax scaling
clf_softmax = train_classifier(X_train, y_train)

#%% Calculate accuracy after softmax scaling
accuracy_softmax = calculateFmeasure(clf_softmax, X_test, y_test)

#%% Add accuracy after softmax scaling to the bar graph data
data_after.append(accuracy_softmax)

#%% Apply max absolute scaling
maxabs_scaler = MaxAbsScaler()
X_maxabs = maxabs_scaler.fit_transform(X)

#%% split the data for training and testing (for maxabs scaling)
X_train, X_test, y_train, y_test= trainSplit(X_maxabs, y)

#%% Train the classifier after max absolute scaling
clf_maxabs = train_classifier(X_train, y_train)

#%% Calculate accuracy after max absolute scaling
accuracy_maxabs = calculateFmeasure(clf_maxabs, X_test, y_test)

#%% Add accuracy after max absolute scaling to the bar graph data
data_after.append(accuracy_maxabs)

#%% Apply robust scaling
robust_scaler = RobustScaler()
X_robust = robust_scaler.fit_transform(X)

#%% split the data for training and testing (for robust scaling)
X_train, X_test, y_train, y_test= trainSplit(X_robust, y)

#%% Train the classifier after robust scaling
clf_robust = train_classifier(X_train, y_train)

#%% Calculate accuracy after robust scaling
accuracy_robust = calculateFmeasure(clf_robust, X_test, y_test)

#%% Add accuracy after robust scaling to the bar graph data
data_after.append(accuracy_robust)

#%% Create the bar graph
create_bar_graph(data_labels, data_before, data_after)
