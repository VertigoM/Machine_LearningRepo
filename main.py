import time

from word_complexity_predictor import WordComplexityPredictor

TEST_FILE = 'data/test.xlsx'

if __name__ == '__main__':
    start_time = time.time()
    wd = WordComplexityPredictor(debug=True)
    wd.self_analyze()
    print("--- %s seconds---" % (time.time() - start_time))
