import sys

sys.path.insert(0, '../source')
import get_data
from LSTM_model import LSTM_model

# Features dimensions
TEXT_DIM = 53
LIWC_DIM = 64
LDA_DIM = 50
LOCATION_DIM = 886
MEDIA_DIM = 1000

learning_rate = 0.001
display_step = 1000
input_dimension = TEXT_DIM + LIWC_DIM + LDA_DIM + LOCATION_DIM + MEDIA_DIM  # number of features


def main(args):
    #  arg[0]=n_hidden; arg[1]=windows_size; arg[2]=learning_rate; arg[3]=1/0 multi_layer
    labels = [0, 1, 2, 3]
    n_hidden = int(args[0])
    windows_size = int(args[1])
    learning_rate = float(args[2])
    if int(args[3]) == 1:
        multi_layer = True
    else:
        multi_layer = False
    training_iters = [5000, 10000, 15000]
    batch_sizes = [16, 32, 64, 128]

    model = LSTM_model(learning_rate=learning_rate,
                       n_hidden=n_hidden,
                       input_dimension=input_dimension,
                       output_dimension=2,
                       n_input=windows_size,
                       multi_layer=multi_layer)

    for label in labels:
        for training_iter in training_iters:
            for batch_size in batch_sizes:
                train_input, test_input, train_output, test_output = \
                    get_data.get_train_test_windows(windows_size, label)
                model.update_params(training_iters=training_iter, batch_size=batch_size, label=label)
                model.train(train_input, train_output, batch_size, test_input, test_output)


if __name__ == "__main__":
    main(sys.argv[1:])
