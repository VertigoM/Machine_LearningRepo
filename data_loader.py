import numpy as np
import pandas as pd


class DataLoader:
    __instance = None

    @staticmethod
    def get_instance():
        if DataLoader.__instance is None:
            DataLoader()
        return DataLoader.__instance

    def __init__(self):
        self.__age_of_acquisition = self.load_aoq()
        self.__concreteness = self.load_concreteness()
        self.__unigrams_list = self.load_unilist()

        DataLoader.__instance = self

    @staticmethod
    def load_aoq():
        dtypes = {0: str, 1: np.float64, 2: np.float64, 3: np.float64, 4: np.float64, 5: np.float64, 6: np.float64}
        csv_data = pd.read_csv('data/age_of_acquisition.csv', header=None,
                               encoding='unicode-escape', on_bad_lines='skip', index_col=0,
                               dtype=dtypes, na_values='NA')
        csv_data = csv_data.replace(np.nan, 0)
        c_dict = csv_data.T.to_dict('list')
        return c_dict

    @staticmethod
    def load_concreteness():
        dtypes = {0: str, 1: np.float64, 2: np.float64, 3: np.float64, 4: np.float64, 5: np.float64, 6: np.float64,
                  7: np.float64, 8: str}
        csv_data = pd.read_csv('data/concreteness.csv', header=None,
                               encoding='unicode-escape', on_bad_lines='skip', index_col=0,
                               dtype=dtypes, usecols=range(8))
        csv_data = csv_data.replace(np.nan, 0)
        c_dict = csv_data.T.to_dict('list')
        return c_dict

    @staticmethod
    def load_unilist():
        dtypes = {0: str, 1: str, 2: np.float64}
        csv_data = pd.read_csv('data/unigrams_list.csv', header=None,
                               encoding='unicode-escape', on_bad_lines='skip', index_col=0, dtype=dtypes)
        csv_data = csv_data.dropna()
        c_dict = csv_data.T.to_dict('list')
        return c_dict

    def get_word_data_aoq(self, word):
        data = self.__age_of_acquisition.get(word, [0, 0, 0, 0, 0, 0])
        return data

    def get_word_data_concreteness(self, word):
        data = self.__concreteness.get(word, [0, 0, 0, 0, 0, 0, 0])[:7]
        return data

    def get_word_data_unigrams_word(self, word):
        data = self.__unigrams_list.get(word)
        if data is None:
            return [0]
        return [data[1]]

    def get_word_data_unigrams_lemma(self, lemma):
        data = self.__unigrams_list.get(lemma)
        if data is None:
            return [0]
        return [data[1]]
