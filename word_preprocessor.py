import data_loader

import nltk
import numpy as np
from dale_chall import DaleChallSet
from nltk.corpus import wordnet
from sklearn.preprocessing import Normalizer
from data_loader import DataLoader


class WordPreprocessor:
    def __init__(self):
        self._lemmatizer = nltk.WordNetLemmatizer()
        self._data_loader = DataLoader.get_instance()
        self.__dale_chall_set = DaleChallSet()
        self.__normalizer = Normalizer()

    def get_feature_vector(self, word):
        lword = word.lower()
        lemma = self._lemmatizer.lemmatize(word)

        word_features = [len(lemma)]
        word_features.extend(self._get_age_of_acquisition(lword))
        word_features.extend(self._get_concreteness_score(lword))
        word_features.extend(self._get_unigrams_count(lword))
        word_features.extend(self._get_unilem_count(lemma))
        word_features.append(self.__is_dale_chall(lword))
        word_features.append(self.__is_title(word))
        word_features.append(self.__get_number_of_synsets(lword))
        word_features.append(self.__get_number_of_hyponyms(lword))

        return np.array([word_features])

    def __is_dale_chall(self, word):
        is_dale_chall = word in self.__dale_chall_set.DALE_CHALL_SET
        return int(is_dale_chall)

    def __normalize_data(self, x):
        self.__normalizer.fit([x])
        return self.__normalizer.transform(x)

    @staticmethod
    def __is_title(word):
        is_title = word[0].isupper()
        return int(is_title)

    @staticmethod
    def __get_number_of_synsets(word):
        synsets = wordnet.synsets(word)
        return len(synsets)

    @staticmethod
    def __get_number_of_hyponyms(word):
        synsets = wordnet.synsets(word)
        h_set = set()
        for synset in synsets:
            h_set.update(synset.hyponyms())
        return len(h_set)

    def _get_age_of_acquisition(self, word):
        return self._data_loader.get_word_data_aoq(word)

    def _get_concreteness_score(self, word):
        return self._data_loader.get_word_data_concreteness(word)

    def _get_unigrams_count(self, word):
        return self._data_loader.get_word_data_unigrams_word(word)

    def _get_unilem_count(self, lemma):
        return self._data_loader.get_word_data_unigrams_lemma(lemma)
