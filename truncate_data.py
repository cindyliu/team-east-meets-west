TRAIN_DATA_IN = 'trainingData.txt'
TRAIN_DATA_OUT = 'trainingDataSubSet.txt'
TRAIN_TRUTH_IN = 'trainingTruth.txt'
TRAIT_TRUTH_OUT = 'trainingTruthSubSet.txt'
TEST_DATA_IN = 'testData.txt'
TEST_DATA_OUT = 'testDataSubSet.txt'
ROWS_TO_COPY = 5000

data_in = [TRAIN_DATA_IN, TRAIN_TRUTH_IN, TEST_DATA_IN]
data_out = [TRAIN_DATA_OUT, TRAIT_TRUTH_OUT, TEST_DATA_OUT]

for i, data_set in enumerate(data_in):
    with open(data_set, 'r') as inf:
        with open(data_out[i], 'w') as outf:
            for row in range(ROWS_TO_COPY):
                outf.write(inf.readline())
