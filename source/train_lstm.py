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


def main(label):
    windows_sizes = [2, 3, 4, 5, 6]
    is_multi_layer = [False, True]
    n_hiddens = [256, 512]
    label = int(label[0])
    training_iters = [5000, 10000, 15000]
    batch_sizes = [7, 14, 21]
    for windows_size in windows_sizes:
        for multi_layer in is_multi_layer:
            for n_hidden in n_hiddens:
                for training_iter in training_iters:
                    for batch_size in batch_sizes:
                        train_input, test_input, train_output, test_output = \
                            get_data.get_train_test_windows(windows_size, label)
                        model = LSTM_model(learning_rate=learning_rate, training_iters=training_iter,
                                           display_step=display_step, input_dimension=input_dimension,
                                           output_dimension=2, n_hidden=n_hidden,
                                           n_input=windows_size, batch_size=batch_size, multi_layer=multi_layer,
                                           label=label)
                        model.train(train_input, train_output, batch_size, test_input, test_output)


if __name__ == "__main__":
    main(sys.argv[1:])
