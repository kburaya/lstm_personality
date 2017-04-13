from pymongo import MongoClient
import numpy as np

###
MONGO_PORT = 27017
MONGO_HOST = '172.29.29.24'
###


def connect_to_database_books_collection(host, port, db_name):
    # connects to mongo
    client = MongoClient(host, port)
    return client[db_name]


def get_period_data(n_periods, modality, db):
    # get period data from db
    # return period data, where all the users have the data in every period
    # the format for name of period collections in db: [modality]_[period_number]_features
    periods_data = dict()
    db = connect_to_database_books_collection(MONGO_HOST, MONGO_PORT, db)
    for period_number in range(1, n_periods + 1):
        collection = '%s_%d_features' % (modality, period_number)
        period_users = db[collection].find()
        for user in period_users:
            if user['_id'] not in periods_data:
                periods_data[user['_id']] = list()
            user_features = list()
            for feature in user:
                if feature != '_id':
                    user_features.append(user[feature])
            periods_data[user['_id']].append(user_features)

        output = dict()
        for user in periods_data:
            if len(periods_data[user]) == n_periods:
                output[user] = periods_data[user]

    return output


def input_output_generation(period_features, groundtruth_collection_name, db, mbti_position):
    # get data from mongo and prepare it for neural network format
    db = connect_to_database_books_collection(MONGO_HOST, MONGO_PORT, db)
    # mbti_position is what letter in mbti profile we want to predict
    # outout_data will be 1/0
    output_data = list()
    #input data - list of lists of 53-dimensional vectors, one for each period
    input_data = list()

    for user in period_features:
        user_data = db[groundtruth_collection_name].find_one({'twitterUserName': user})
        user_features = list()
        for features in period_features[user]:
            user_features.append(features)
        input_data.append(user_features)
        output_data.append(convert_mbti_to_vector(user_data['mbti'], mbti_position))

    return input_data, output_data


def convert_mbti_to_vector(mbti, mbti_position):
    #convert mbti to 4-dimensional vector representation
    # I S T J -> 0
    # E N F P -> 1
    mbti_vector = list()
    if mbti[0] == 'I': mbti_vector.append(0)
    if mbti[0] == 'E': mbti_vector.append(1)
    if mbti[1] == 'S': mbti_vector.append(0)
    if mbti[1] == 'N': mbti_vector.append(1)
    if mbti[2] == 'T': mbti_vector.append(0)
    if mbti[2] == 'F': mbti_vector.append(1)
    if mbti[3] == 'J': mbti_vector.append(0)
    if mbti[3] == 'P': mbti_vector.append(1)

    output = mbti_vector[mbti_position]
    output_vector = [output]

    if output == 0: output_vector.append(1)
    else: output_vector.append(0)

    return output_vector


def split_data_to_train_test(input_data, output_data, threshold = 0.8):
    random_state = np.random.rand(len(input_data))

    input_train, output_train, input_test, output_test = list(), list(), list(), list()

    for i in range(1, len(random_state)):
        if random_state[i] < threshold:
            input_train.append(input_data[i])
            output_train.append(output_data[i])
        else:
            input_test.append(input_data[i])
            output_test.append(output_data[i])

    return input_train, output_train, input_test, output_test

# period_data = get_period_data(2, 'tweets', 'tweets_10_periods')
# input_data, output_data = input_output_generation(period_data, 'users', 'user-profiling', 1)
# train_input, train_output, test_input, test_output = split_data_to_train_test(input_data, output_data)