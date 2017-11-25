# -*- coding:utf-8 -*-
import os
import json
import pickle
from random import randint

from pyvi.pyvi import ViTokenizer
from gensim import corpora, matutils
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

SPECIAL_CHARACTER = '0123456789%@$.‘​,“”’•…™=+-!;/()*"&^:#|\n\t\''

class FileReader(object):
    def __init__(self, filePath):
        self.filePath = filePath

    def read(self):
        with open(self.filePath) as f:
            s = f.read()
        return s

    def content(self):
        s = self.read()
        return s

    def read_json(self):
        with open(self.filePath) as f:
            s = json.loads(f.read().decode("utf-8", errors="ignore"))
        return s

    def read_stopwords(self):
        with open(self.filePath, 'r') as f:
            stopwords = set([w.strip().replace(' ', '_') for w in f.readlines()])
        return stopwords

    def load_dictionary(self):
        return corpora.Dictionary.load_from_text(self.filePath)

class FileStore(object):
    def __init__(self, filePath, data = None):
        self.filePath = filePath
        self.data = data

    def store_json(self):
        with open(self.filePath, 'w') as outfile:
            json.dump(self.data, outfile, ensure_ascii=False)

    def store_dictionary(self, dict_words):
        dictionary = corpora.Dictionary(dict_words)
        dictionary.filter_extremes(no_below=20, no_above=0.3)
        dictionary.save_as_text(self.filePath)

    def save_pickle(self,  obj):
        with open(self.filePath, 'wb') as file:
            pickle.dump(obj, file)

class DataLoader(object):
    def __init__(self, path):
        self.path = path

    def _get_files_of_category(self):
        files = {}
        categories = os.listdir(self.path)
        for folder in categories:
            if os.path.isdir(self.path + folder):
                files[folder] = [self.path + folder + "/" + file for file in os.listdir(self.path + folder)]
        self.files = files
    def _get_json(self):
        self._get_files_of_category()
        data = []
        for cate in self.files:
            rand = 1000
            i = 0
            for file in self.files[cate]:
                content = FileReader(file).content()
                data.append({
                    "category": cate,
                    "content": content
                })
                if i == rand:
                    break
                else:
                    i += 1
        return data

class NLP(object):
    def __init__(self, text = None):
        self.text = text
        self.__set_stopwords()

    # Lấy stopwords từ file stopwords-vi.txt
    def __set_stopwords(self):
        self.stopwords = FileReader("stopwords-vi.txt").read_stopwords()

    # Tách từ sử dụng thư viện ViTokenizer
    def segmentation(self):
        return ViTokenizer.tokenize(self.text)

    # Loại bỏ các ký tự đặc biệt
    def split_words(self):
        text = self.segmentation()
        try:
            return [x.strip(SPECIAL_CHARACTER.decode("utf-8")).lower() for x in text.split()]
        except TypeError:
            return []

    # Loại bỏ stop words, Lấy feature words
    def get_words_feature(self):
        split_words = self.split_words()
        return [word for word in split_words if word.encode('utf-8') not in self.stopwords]

class FeatureExtraction(object):
    def __init__(self, data):
        self.data = data

    def build_dictionary(self):
        print 'Building dictionary'
        dict_words = []
        i = 0
        for text in self.data:
            i += 1
            print "Build Dict Step {} / {}".format(i, len(self.data))
            words = NLP(text=text['content']).get_words_feature()
            dict_words.append(words)
        FileStore(filePath="dictionary.txt").store_dictionary(dict_words)

    def load_dictionary(self):
        if os.path.exists("dictionary.txt") == False:
            self.build_dictionary()
        self.dictionary = FileReader("dictionary.txt").load_dictionary()

    def get_dense(self, text):
        self.load_dictionary()
        words = NLP(text).get_words_feature()
        # Bag of words
        vec = self.dictionary.doc2bow(words)
        dense = list(matutils.corpus2dense([vec], num_terms=len(self.dictionary)).T[0])
        return dense

    def __build_dataset(self):
        self.features = []
        self.labels = []
        i = 0
        for d in self.data:
            i += 1
            print "Build Dataset Step {} / {}".format(i, len(self.data))
            self.features.append(self.get_dense(d['content']))
            self.labels.append(d['category'])

    def get_data_and_label(self):
        self.__build_dataset()
        return self.features, self.labels

class Classifier_SVM(object):
    def __init__(self, features_train = None, labels_train = None, estimator = LinearSVC(random_state=0)):
        self.features_train = features_train
        self.labels_train = labels_train
        self.estimator = estimator

    def training(self):
        self.estimator.fit(self.features_train, self.labels_train)
        # self.__training_result()

    def save_model(self, filePath, obj):
        FileStore(filePath=filePath).save_pickle(obj)

    # def __training_result(self):
    #     y_true, y_pred = self.labels_test, self.estimator.predict(self.features_test)
    #     print("SVM\n" + classification_report(y_true, y_pred))

class Classifier_NB(object):
    def __init__(self, features_train=None, labels_train=None, estimator=MultinomialNB()):
        self.features_train = features_train
        self.labels_train = labels_train
        self.estimator = estimator

    def training(self):
        self.estimator.fit(self.features_train, self.labels_train)
        # self.__training_result()

    def save_model(self, filePath, obj):
        FileStore(filePath=filePath).save_pickle(obj)

    # def __training_result(self):
    #     y_true, y_pred = self.labels_test, self.estimator.predict(self.features_test)
    #     print("NB\n" + classification_report(y_true, y_pred))

class Classifier_KNN(object):
    def __init__(self, features_train=None, labels_train=None, estimator=KNeighborsClassifier(n_neighbors=13)):
        self.features_train = features_train
        self.labels_train = labels_train
        self.estimator = estimator

    def training(self):
        self.estimator.fit(self.features_train, self.labels_train)
        # self.__training_result()

    def save_model(self, filePath, obj):
        FileStore(filePath=filePath).save_pickle(obj)

    # def __training_result(self):
    #     y_true, y_pred = self.labels_test, self.estimator.predict(self.features_test)
    #     print("K-NN\n" + classification_report(y_true, y_pred))

def report(features_test, labels_test, estimator):
    y_true, y_pred = labels_test, estimator.predict(features_test)
    print(classification_report(y_true, y_pred))

json_test = DataLoader("Data/Test/")._get_json()
FileStore("test.json", data=json_test).store_json()
# train_json = DataLoader("Data/Train/")._get_json()
# FileStore("train.json", train_json).store_json()

# train_json = FileReader("train.json").read_json()
test_json = FileReader("test.json").read_json()

# features_train, labels_train = FeatureExtraction(data=train_json).get_data_and_label()
features_test, labels_test = FeatureExtraction(data=test_json).get_data_and_label()

# nb = Classifier_NB(features_train=features_train, labels_train=labels_train)
# nb.training()
# nb.save_model("nb_model.pkl", obj=nb.estimator)
#
# svm = Classifier_SVM(features_train=features_train, labels_train=labels_train)
# svm.training()
# svm.save_model("svm_model.pkl", obj=svm.estimator)

# knn = Classifier_KNN(features_train=features_train, labels_train=labels_train)
# knn.training()
# knn.save_model("knn_model.pkl", obj=knn.estimator)

# est = Classifier(features_train=features_train, features_test=features_test, labels_train=labels_train, labels_test=labels_test)
# est.training()
# est.save_model(filePath='linear_svc_model.pkl', obj=est.estimator)

# x = FeatureExtraction(data="").get_dense(FileReader("Data/Test/giai_tri/giai-tri_3.txt").content().decode("utf-8"))
with open("svm_model.pkl", 'rb') as file:
    estimator_svm = pickle.load(file)
print "SVM"
report(features_test, labels_test, estimator_svm)

with open("nb_model.pkl", 'rb') as file:
    estimator_nb = pickle.load(file)
print "NB"
report(features_test, labels_test, estimator_nb)