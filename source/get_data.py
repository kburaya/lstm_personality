from pymongo import MongoClient
import numpy as np
import random
from sklearn.decomposition import NMF
import logging
import hashlib
import os
import pickle
import uuid
from imblearn.over_sampling import SMOTE

###
# Mongo params
MONGO_PORT = 27017
MONGO_HOST = 'localhost'
MONGO_DB = 'mbti_research'
MONGO_DB_RESTORE = MONGO_DB + '_restore'
# Features dimensions
TEXT_DIM = 53
LIWC_DIM = 64
LDA_DIM = 50
LOCATION_DIM = 886
MEDIA_DIM = 1000
PERIODS = 10
###
features_dim = TEXT_DIM + LIWC_DIM + LDA_DIM + LOCATION_DIM
logging.basicConfig(filename='%s.log' % str(uuid.uuid4()), filemode='w', level=logging.DEBUG, format='%(asctime)s %(message)s')
hash_maker = hashlib.md5()


def connect_to_database(host, port, db_name):
    # connects to mongo
    client = MongoClient(host, port)
    return client[db_name]


def get_labels_stats(train_output, test_output, mbti_position):
    # train_set
    train_zero, train_one = 0, 0
    for train_label in train_output:
        if train_label == [1, 0]:
            train_zero += 1
        else:
            train_one += 1
    logging.info('Stats for %s/%s label in train set. Number: %d/%d, percentage: %.2f, %.2f' %
                 (get_label_letter(mbti_position, 0), get_label_letter(mbti_position, 1),
                  train_zero, train_one,
                  train_zero/(train_zero + train_one), train_one/(train_zero + train_one)))

    # train_set
    test_zero, test_one = 0, 0
    for test_label in test_output:
        if test_label == [1, 0]:
            test_zero += 1
        else:
            test_one += 1
    logging.info('Stats for %s/%s label in test set. Number: %d/%d, percentage: %.2f, %.2f' %
                 (get_label_letter(mbti_position, 0), get_label_letter(mbti_position, 1),
                  test_zero, test_one,
                  test_zero / (test_zero + test_one), test_one / (test_zero + test_one)))


def download_window_data(window_size, mbti_position):
    try:
        train_input = pickle.load(open('../store/window_%d/train_input_%d.pkl' % (window_size, mbti_position), 'rb'))
        train_output = pickle.load(open('../store/window_%d/train_output_%d.pkl' % (window_size, mbti_position), 'rb'))
        test_input = pickle.load(open('../store/window_%d/test_input_%d.pkl' % (window_size, mbti_position), 'rb'))
        test_output = pickle.load(open('../store/window_%d/test_output_%d.pkl' % (window_size, mbti_position), 'rb'))
        logging.info('Found {%d} train samples, {%d} test samples for {%d} period-window' %
                     (len(train_input), len(test_output), window_size))
        get_labels_stats(train_output, test_output, mbti_position)
        return train_input, test_input, train_output, test_output
    except:
        raise FileNotFoundError('There is no prepared window data!')


def get_period_data(n_periods, features_types, db, features_dim):
    # get period data from db
    # return period data, where all the users have the data in every period
    # the format for name of period collections in db: [modality]_[period_number]_[features_type]
    periods_data = dict()
    db = connect_to_database(MONGO_HOST, MONGO_PORT, db)
    for period_number in range(1, n_periods + 1):
        logging.info('Getting data for {%d} period in begin' % period_number)
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
    logging.info("Find {%d} users in {%d} periods" % (len(output), n_periods))
    return output


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


def get_label_letter(mbti_position, label):
    mbti = ['EI', 'NS', 'FT', 'PJ']
    return mbti[mbti_position][label]


def split_data_to_train_test():
    # split ALL users (table: users) into train(0.8) and test(0.2) sets based on the distribution of their MBTI
    db = connect_to_database(MONGO_HOST, MONGO_PORT, MONGO_DB_RESTORE)
    mbti_types = db['users'].distinct('mbti')
    for mbti in mbti_types:
        users = db['users'].find({'mbti': mbti})
        randoms = list()
        for i in range(users.count()):
            randoms.append(random.uniform(0, 1))
        for (user, p) in zip(users, randoms):
            if p < 0.8:
                user['set'] = 'train'
            else:
                user['set'] = 'test'
            db['users'].update({'_id': user['_id']}, user)


def get_batch(input_data, output_data, batch_size, offset):
    if offset + batch_size >= len(output_data):
        batch_size = len(output_data) - offset - 1
    indexes = range(offset, offset + batch_size)
    input_batch, output_batch = list(), list()

    for i in indexes:
        input_batch.append(input_data[i])
        output_batch.append(output_data[i])

    return input_batch, output_batch


def fill_missed_modality(period):
    # get all users with activity in current period
    # build the matrix and do it factorization
    # need to know the order of features to write them into database then
    # store all features in one vector as a result
    db = connect_to_database(MONGO_HOST, MONGO_PORT, MONGO_DB)
    modalities = ['text', 'liwc', 'lda', 'location', 'media']
    modalities_dimensions = [53, 64, 50, 886, 1000]
    period_users = list()  # all active users in any social network for this period
    for modality in modalities:
        users = db['MBTI_%d_%s' % (period, modality)].distinct('_id')
        for user in users:
            if user not in period_users:
                period_users.append(user)

    users_features = list()  # the order of users will be the same as in users list
    incomplete_dimensions = 0
    for user in period_users:
        feature_vector = list()
        for modality, dimension in zip(modalities, modalities_dimensions):
            features = db['MBTI_%d_%s' % (period, modality)].find_one({"_id": user})
            if features is None:
                feature_vector.extend([0] * dimension)
                incomplete_dimensions += 1
            else:
                for feature_name in features:
                    if feature_name != '_id':
                        if features[feature_name] < 0:  # FIXME it happens only for sentiment score here
                            feature_vector.append((-1.0) * features[feature_name])
                        else:
                            feature_vector.append(features[feature_name])
        users_features.append(np.array(feature_vector))

    logging.info("Found {%d} incomplete dimensions" % incomplete_dimensions)
    R = np.array(users_features)
    model = NMF(init='random', random_state=0)
    W = model.fit_transform(R)
    H = model.components_
    transformed_data = np.dot(W, H)

    db_restore = connect_to_database(MONGO_HOST, MONGO_PORT, MONGO_DB_RESTORE)
    for user in period_users:
        sum_dimension = 0
        user_position = period_users.index(user)
        for modality, dimension in zip(modalities, modalities_dimensions):
            features = db['MBTI_%d_%s' % (period, modality)].find_one({'_id': user})
            if features is None:
                features_to_db = dict()
                features_to_db['_id'] = user
                for j in range(0, dimension):
                    features_to_db['%s_%d' % (modality, j + 1)] = \
                        transformed_data[user_position][sum_dimension + j]
                db_restore['MBTI_%d_%s' % (period, modality)].insert(features_to_db)
                logging.info("Modality {%s} was restore for user {%s}" % (modality, user))

            sum_dimension += dimension


def store_modalities_in_one_vector(modalities, period):
    db = connect_to_database(MONGO_HOST, MONGO_PORT, MONGO_DB_RESTORE)
    users = db['MBTI_%d_%s' % (period, modalities[0])].distinct('_id')
    for user in users:
        user_vector = dict()
        for modality in modalities:
            modality_vector = db['MBTI_%d_%s' % (period, modality)].find_one({'_id': user})
            del modality_vector['_id']
            for feature in modality_vector:
                user_vector["%s_%s" % (feature, modality)] = modality_vector[feature]
        user_vector['_id'] = user
        db['period_%d' % period].insert(user_vector)


def input_output_generation(train_input, test_input, train_output, test_output, n_periods):
    # get data from train/test inputs and prepare it for LSTM format
    # input data - list of lists of vectors, one for each period
    train_i, test_i = list(), list()
    train_o, test_o = list(), list()
    users_num = 0

    logging.info('Generation train/test input/output')
    for user in train_input:
        train_i.append(list())
        for period in range(0, n_periods):
            train_i[len(train_i) - 1].append(train_input[user][period + 1])
        train_o.append(train_output[user])
        users_num += 1

    users_num = 0
    for user in test_input:
        test_i.append(list())
        for period in range(0, n_periods):
            test_i[users_num].append(test_input[user][period + 1])
        test_o.append(test_output[user])
        users_num += 1

    return train_i, test_i, train_o, test_o


def get_train_test_windows(n_periods, mbti_position, store=True):
    try:
        return download_window_data(n_periods, mbti_position)
    except FileNotFoundError:
        logging.info("There is no prepared data for window-periods")
    # find users, that posts in every window of [n_periods] periods
    # 1 2 3 4 -> from this will return all users that posts in [1, 2], [2, 3].
    # will give md5(user_id) for every user to avoid the duplication of ids
    # will return test/train sets for period windows
    db = connect_to_database(MONGO_HOST, MONGO_PORT, MONGO_DB_RESTORE)
    users_mapping = dict()  # map md5(user) -> user
    train_output, test_output = dict(), dict()  # dict md5(user) -> MBTI_vector
    train_input, test_input = dict(), dict()  # dict md5(user) -> dict period -> features
    logging.info('Begin to collect data for %d-windows' % n_periods)
    # get all users
    for i in range(1, PERIODS):
        if i + n_periods > PERIODS + 1:
            break
        logging.info('Getting data for %d period in the beginning' % i)
        window_users = dict()
        for p in range(0, n_periods):
            period_users = db['period_%d' % (i + p)].find()
            for user in period_users:
                if user['_id'] not in window_users:
                    window_users[user['_id']] = 1
                else:
                    window_users[user['_id']] += 1

        for user in window_users:
            if window_users[user] != n_periods:
                continue

            new_id = str(uuid.uuid4())
            users_mapping[new_id] = user
            user = db['users'].find_one({'twitterUserName': user})
            if user is None:
                logging.error('Find None user %s' % user)
                continue

            if user['set'] == 'train':
                train_input[new_id] = dict()
                train_output[new_id] = dict()
                window_period = 1
                for p in range(i, i + n_periods):
                    features = db['period_%d' % p].find_one({'_id': user['twitterUserName']})
                    train_input[new_id][window_period] = list()
                    for feature in features:
                        if feature != '_id':
                            train_input[new_id][window_period].append(features[feature])
                    window_period += 1
                train_output[new_id] = convert_mbti_to_vector(user['mbti'], mbti_position)
            else:
                test_input[new_id] = dict()
                test_output[new_id] = dict()
                window_period = 1
                for p in range(i, i + n_periods):
                    features = db['period_%d' % p].find_one({'_id': user['twitterUserName']})
                    test_input[new_id][window_period] = list()
                    for feature in features:
                        if feature != '_id':
                            test_input[new_id][window_period].append(features[feature])
                    window_period += 1
                test_output[new_id] = convert_mbti_to_vector(user['mbti'], mbti_position)

    logging.info("Found {%d} train samples, {%d} test samples for {%d} period-window" %
                 (len(train_input), len(test_output), n_periods))

    train_input, test_input, train_output, test_output = input_output_generation(train_input, test_input,
                                                                                 train_output, test_output, n_periods)
    get_labels_stats(train_output, test_output, mbti_position)
    if store:
        logging.info('Saving data...')
        if not os.path.exists('../store/window_%d' % n_periods):
            os.makedirs('../store/window_%d' % n_periods)
        pickle.dump(train_input, open('../store/window_%d/train_input_%d.pkl' % (n_periods, mbti_position), 'wb'))
        pickle.dump(train_output, open('../store/window_%d/train_output_%d.pkl' % (n_periods, mbti_position), 'wb'))
        pickle.dump(test_input, open('../store/window_%d/test_input_%d.pkl' % (n_periods, mbti_position), 'wb'))
        pickle.dump(test_output, open('../store/window_%d/test_output_%d.pkl' % (n_periods, mbti_position), 'wb'))
        logging.info('Data successfully saved')

    return train_input, test_input, train_output, test_output


def apply_oversampling(train_input, test_input, train_output, test_output):
    sm = SMOTE(random_state=42)
    train_input_s, train_output_s = sm.fit_sample(train_input, train_output)
    test_input_s, test_output_s = sm.fit_sample(test_input, test_output)
    return train_input_s, test_input_s, train_output_s, test_output_s


def transforn_outputs_to_list(output):
    # transforms 2-dimensional mbti vector to 1-dimensional for comparison in TF
    result = list()
    for i in range(0, len(output)):
        if output[i][1] == 1:
            result.append(True)
        else:
            result.append(False)
    return result


def transform_int_to_bool(output):
    result = list()
    for i in range(0, len(output)):
        if output[i] == 1:
            result.append(True)
        else:
            result.append(False)
    return result

def main(args):
    periods = args
    for period in periods:
        fill_missed_modality(period)
        store_modalities_in_one_vector(['text', 'liwc', 'lda', 'location', 'media'], period)


# if __name__ == "__main__":
#     # main([2, 3, 4, 5, 6, 7, 8, 9, 10])
#     # split_data_to_train_test()
#     get_train_test_windows(2, 0)

