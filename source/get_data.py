from pymongo import MongoClient
import numpy as np
import numpy
import random
from sklearn.decomposition import NMF

###
MONGO_PORT = 27017
MONGO_HOST = 'localhost'
MONGO_DB = 'mbti_research'
###
# Features dimensions
TEXT_DIM = 53
LIWC_DIM = 64
LDA_DIM = 50
LOCATION_DIM = 886
#
features_dim = TEXT_DIM + LIWC_DIM + LDA_DIM + LOCATION_DIM


def connect_to_database(host, port, db_name):
    # connects to mongo
    client = MongoClient(host, port)
    return client[db_name]


def get_period_data(n_periods, features_types, db, features_dim):
    # get period data from db
    # return period data, where all the users have the data in every period
    # the format for name of period collections in db: [modality]_[period_number]_[features_type]
    periods_data = dict()
    db = connect_to_database(MONGO_HOST, MONGO_PORT, db)
    for period_number in range(1, n_periods + 1):
        period_flags = list()
        for features_type in features_types:
            collection = 'MBTI_%d_%s' % (period_number, features_type)
            period_users = db[collection].find()
            for user in period_users:
                if user['_id'] not in periods_data:
                    periods_data[user['_id']] = list()
                user_features = list()
                for feature in user:
                    if feature != '_id':
                        user_features.append(user[feature])
                if user['_id'] not in period_flags:
                    periods_data[user['_id']].append(list())
                    period_flags.append(user['_id'])
                periods_data[user['_id']][len(periods_data[user['_id']]) - 1].extend(user_features)

    output = dict()
    for user in periods_data:
        if len(periods_data[user]) == n_periods:
            add = True
            for period_features in periods_data[user]:
                if len(period_features) != features_dim:
                    add = False
            if add:
                output[user] = periods_data[user]
    print("Find {%d} users in {%d} periods" % (len(output), n_periods))
    return output


def input_output_generation(period_features, n_periods, groundtruth_collection_name, db, mbti_position):
    # get data from mongo and prepare it for neural network format
    db = connect_to_database(MONGO_HOST, MONGO_PORT, db)
    # mbti_position is what letter in mbti profile we want to predict
    # outout_data will be 1/0
    output_data = list()
    # input data - list of lists of 53-dimensional vectors, one for each period
    input_data = list()
    for p in range(0, n_periods):
        input_data.append(list())

    for period in range(0, n_periods):
        for user in period_features:
            input_data[period].append(period_features[user][period])
    for user in period_features:
        user_data = db[groundtruth_collection_name].find_one({'twitterUserName': user})
        output_data.append(convert_mbti_to_vector(user_data['mbti'], mbti_position))

    return input_data, output_data


def convert_mbti_to_vector(mbti, mbti_position):
    # convert mbti to 4-dimensional vector representation
    # I S T J -> 0
    # E N F P -> 1
    mbti_vector = list()
    if mbti[0] == 'I':
        mbti_vector.append([0, 1])
    if mbti[0] == 'E':
        mbti_vector.append([1, 0])
    if mbti[1] == 'S':
        mbti_vector.append([0, 1])
    if mbti[1] == 'N':
        mbti_vector.append([1, 0])
    if mbti[2] == 'T':
        mbti_vector.append([0, 1])
    if mbti[2] == 'F':
        mbti_vector.append([1, 0])
    if mbti[3] == 'J':
        mbti_vector.append([0, 1])
    if mbti[3] == 'P':
        mbti_vector.append([1, 0])

    return mbti_vector[mbti_position]


def split_data_to_train_test(input_data, output_data, n_periods, threshold=0.9):
    random_state = np.random.rand(len(input_data[0]))

    input_train, output_train, input_test, output_test = list(), list(), list(), list()

    for i in range(1, len(random_state)):
        period_features = list()
        for p in range(0, n_periods):
            period_features.append(input_data[p][i])
        if random_state[i] < threshold:
            input_train.append(period_features)
            output_train.append(output_data[i])
        else:
            input_test.append(period_features)
            output_test.append(output_data[i])

    return input_train, output_train, input_test, output_test


def get_batch(input_data, output_data, batch_size):
    indexes = random.sample(range(0, len(input_data)), batch_size)
    input_batch, output_batch = list(), list()

    for i in indexes:
        input_batch.append(input_data[i])
        output_batch.append(output_data[i])

    return input_batch, output_batch


"""
@INPUT:
    R     : a matrix to be factorized, dimension N x M
    P     : an initial matrix of dimension N x K
    Q     : an initial matrix of dimension M x K
    K     : the number of latent features
    steps : the maximum number of steps to perform the optimisation
    alpha : the learning rate
    beta  : the regularization parameter
@OUTPUT:
    the final matrices P and Q
"""


def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in range(steps):
        if step % 100 == 0:
            print ("{%d} step of NMF" % step)
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - numpy.dot(P[i, :], Q[:, j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = numpy.dot(P, Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i, :], Q[:, j]), 2)
                    for k in range(K):
                        e = e + (beta / 2) * (pow(P[i][k], 2) + pow(Q[k][j], 2))
        if e < 0.001:
            break
    return P, Q.T


def fill_missed_modality(period):
    # get all users with activity in current period
    # build the matrix and do it factorization
    # need to know the order of features to write them into database then
    db = connect_to_database(MONGO_HOST, MONGO_PORT, MONGO_DB)
    modalities = ['text', 'liwc']  # , 'lda', 'location']
    modalities_dimensions = [53, 64]  # , 50, 886]
    period_users = list()  # all active users in any social network for this period
    for modality in modalities:
        users = db['MBTI_%d_%s' % (period, modality)].distinct('_id')
        for user in users:
            if user not in period_users:
                period_users.append(user)

    users_features = list()  # the order of users will be the same as in users list
    incomplete_users = 0
    for user in period_users:
        feature_vector = list()
        for modality, dimension in zip(modalities, modalities_dimensions):
            features = db['MBTI_%d_%s' % (period, modality)].find_one({"_id": user})
            if features is None:
                feature_vector.extend([0] * dimension)
                incomplete_users += 1
            else:
                if len(features) != dimension + 1:
                    print("Error for user {%s}, modality {%s} dimension. Need {%d}, found {%d}"
                          % (user, modality, dimension + 1, len(features)))
                for feature_name in features:
                    if feature_name != '_id':
                        if (features[feature_name] < 0):  # FIXME it happens only for sentiment score here
                            feature_vector.append((-1.0) * features[feature_name])
                        else:
                            feature_vector.append(features[feature_name])
        users_features.append(np.array(feature_vector))

    print("Found {%d} incomplete users" % incomplete_users)
    R = np.array(users_features)
    N = len(users_features)
    M = len(users_features[0])
    K = features_dim

    P = np.random.rand(N, K)
    Q = np.random.rand(M, K)

    nP, nQ = matrix_factorization(R, P, Q, K)
    transformed_data = np.dot(nP, nQ.T)
    # model = NMF(init='random')
    # W = model.fit_transform(R)
    # H = model.components_
    # transformed_data = np.dot(W, H)

    db_restore = connect_to_database(MONGO_HOST, MONGO_PORT, 'mbti_research_restore')
    for user in period_users:
        sum_dimension = 0
        user_position = period_users.index(user)
        for modality, dimension in zip(modalities, modalities_dimensions):
            features = db['MBTI_%d_%s' % (period, modality)].find_one({"_id": user})
            if features is None:
                new_features = transformed_data[user_position][sum_dimension:sum_dimension + dimension]
                features_to_db = dict()
                features_to_db["_id"] = user
                for j in range(0, len(new_features)):
                    features_to_db[str(j)] = new_features[j]
                db_restore['MBTI_%d_%s' % (period, modality)].insert(features_to_db)
            sum_dimension += dimension + 1


fill_missed_modality(period=1)
# period_data = get_period_data(2, 'tweets', 'tweets_10_periods')
# input_data, output_data = input_output_generation(period_data, 2, 'users', 'user-profiling', 1)
# train_input, train_output, test_input, test_output = split_data_to_train_test(input_data, output_data, 2)
