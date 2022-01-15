import time

import pandas as pd
import numpy as np
import pickle

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from word_preprocessor import WordPreprocessor
from sklearn.model_selection import KFold
from sklearn.metrics import balanced_accuracy_score

TRAINING_FILENAME = 'data/train.xlsx'
TRAINED_MODEL = 'model.pickle'


class WordComplexityPredictor:
    def __init__(self, debug=False):
        self.__preprocessor = WordPreprocessor()
        self.__debug = debug

        if not self._load_model():
            # set model as GaussianNB @ https://scikit-learn.org/stable/modules/naive_bayes.html
            self.__model = KNeighborsClassifier(5)
            self._train_and_save()
        self._log("Constructed analyzer.")

    def self_analyze(self):
        start = time.perf_counter()
        kf = KFold(n_splits=10, shuffle=True)

        score_list = []
        x, y = self._compute_data()
        for train_i, test_i in kf.split(x):
            i, j = train_i, test_i
            x_train, x_test = x[i], x[j]
            y_train, y_test = y[i], y[j]

            self.__model.fit(x_train, y_train)
            predictions = self.__model.predict(x_test)
            score = balanced_accuracy_score(y_test, predictions)
            score_list.append(score)
            self._log(f"[INFO] Predicted: {score}")
        self._log(f"[INFO] Average score: {np.mean(score_list)}")
        end = time.perf_counter()
        self._log(f"[INFO] Analyze done in {end - start} seconds")

    def _load_model(self):
        try:
            file = open(TRAINED_MODEL, "rb")
        except FileNotFoundError as e:
            self._log("[INFO]: " + str(e))
            self._log("[INFO]: a new file is being created...")
            return False
        else:
            try:
                self.__model = pickle.load(file)
                self._log("[INFO]: Loaded model from file -", type(self.__model))
                return True
            finally:
                file.close()

    def _train_and_save(self):
        x, y = self._compute_data()
        self._log("[INFO]: data computed for training: ", len(x), len(y))
        self.__model.fit(x, y)
        self._log("[INFO]: Done training data.")

        # python object serialization @ https://docs.python.org/3/library/pickle.html
        try:
            file = open(TRAINED_MODEL, "wb")
        except FileNotFoundError as e:
            self._log("[ERROR]: " + str(e))
        else:
            try:
                pickle.dump(self.__model, file)
                self._log("[INFO]: Trained and saved data to file.")
            finally:
                file.close()

    def _compute_data(self):
        # numpy.ma @ https://numpy.org/doc/stable/reference/maskedarray.generic.html
        # numpy.row_stack @ https://numpy.org/devdocs/reference/generated/numpy.row_stack.html
        if '.csv' in TRAINING_FILENAME:
            dtypes = {'word': str, 'sentence': str, 'index': np.int32, 'label': np.float64}
            training_data = pd.read_csv(TRAINING_FILENAME, dtype=dtypes)
            x = np.ma.row_stack([self.__preprocessor.get_feature_vector(row["word"])
                                 for _, row in training_data.iterrows()])

            y = np.array([row["label"] for _, row in training_data.iterrows()])
        elif '.xlsx' in TRAINING_FILENAME:
            dtypes = {'sentence': str, 'token': str, 'complexity': np.float64}
            training_data = pd.read_excel(TRAINING_FILENAME, dtype=dtypes, keep_default_na=False)
            x = np.ma.row_stack([(self.__preprocessor.get_feature_vector(row['token']))
                                 for _, row in training_data.iterrows()])

            y = np.array([row['complex'] for _, row in training_data.iterrows()])
        else:
            raise Exception('Training file format not fit.')
        return x, y

    def predict(self, word):
        features = self.__preprocessor.get_feature_vector(word)
        predictions = self.__model.predict(features)
        self._log("Predicted {} with complexity {}".format(word, predictions))
        return predictions

    def _log(self, *args):
        # logging HOWTO @ https://docs.python.org/3/howto/logging.html
        if self.__debug:
            for message in args:
                print(message, end='\n')
