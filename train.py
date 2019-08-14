import os
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier

train = []
test = []
y_train = []
y_test = []

listtrain = os.listdir('data/classify_data/train/')
for i in listtrain:
    f = open("data/classify_data/train/"+i, encoding="utf8")
    line = f.readline()
    while(line):
        train.append(line)
        y_train.append(int(i.split('.')[0]))
        line = f.readline()
    f.close()

f = open("data/classify_data/test/data.txt", encoding="utf8")
line = f.readline()
while(line):
    test.append(line)
    line = f.readline()
f.close()
del f

f = open("data/classify_data/test/label.txt", encoding="utf8")
line = f.readline()
while(line):
    y_test.append(int(line))
    line = f.readline()
f.close()
del f

stopwords = []
f = open("stopwords.txt", encoding="utf8")
line = f.readline()
while(line):
    stopwords.append(line)
    line = f.readline()
f.close()
del f
# max_features = [200, 250, 300, 350, 400, 450, 500]
kernels = ['linear']
C_list = [10.0]
n_estimators = 10
# for n_features in tqdm(max_features):
#     for ker in kernels:
#         for c in C_list:
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(train)
X_test = vectorizer.transform(test).toarray()
X_train = X.toarray()
clf = SVC(C=10.0, kernel='linear', probability=True)
clf.fit(X_train, y_train)
print("done")
# print("max_features: {}, C: {}, kernel: {}".format(n_features, c, ker))
print("              {}".format(clf.score(X_test, y_test)))