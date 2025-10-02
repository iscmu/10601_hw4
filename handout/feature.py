import csv
import numpy as np
import argparse

VECTOR_LEN = 300   # Length of glove vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and glove_embeddings.txt

################################################################################
# We have provided you the functions for loading the tsv and txt files. Feel   #
# free to use them! No need to change them at all.                             #
################################################################################


def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An np.ndarray of shape N. N is the number of data points in the tsv file.
        Each element dataset[i] is a tuple (label, review), where the label is
        an integer (0 or 1) and the review is a string.
    """
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8',
                         dtype='l,O')
    return dataset


def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the glove
    embeddings.

    Parameters:
        file (str): File path to the glove embedding file.

    Returns:
        A dictionary indexed by words, returning the corresponding glove
        embedding np.ndarray.
    """
    glove_map = dict()
    with open(file, encoding='utf-8') as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            word, embedding = row[0], row[1:]
            glove_map[word] = np.array(embedding, dtype=float)
    return glove_map

if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the train_input path, you can use `args.train_input`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to training input .tsv file')
    parser.add_argument("validation_input", type=str, help='path to validation input .tsv file')
    parser.add_argument("test_input", type=str, help='path to the input .tsv file')
    parser.add_argument("feature_dictionary_in", type=str, 
                        help='path to the GloVe feature dictionary .txt file')
    parser.add_argument("train_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the training data should be written')
    parser.add_argument("validation_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the validation data should be written')
    parser.add_argument("test_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the test data should be written')
    args = parser.parse_args()

    def convert_features(data_path, features_path):
        data = load_tsv_dataset(data_path)
        features = load_feature_dictionary(features_path)

        output = []
        labels = []

        for i in data:
            label, review = i
            # zero_vec =  np.array([0.0 for i in range(len(next(iter(features.values()))))])
            num_features = len(next(iter(features.values())))
            vec = np.array([0.0 for i in range(num_features)])
            listofwords = review.split()
            cnt = 0
            for w in listofwords:
                if w in features:
                    vec += features.get(w, np.array([0.0 for i in range(num_features)]))
                    cnt += 1
            output.append(vec / cnt)
            labels.append(label)
        return output, labels
    
    def write_o_file(labels, output, file):
        with open(file, 'w') as f:
            for i in range(len(labels)):
                f.write(f"{round(labels[i], 6):.6f}")
                for word in output[i]:
                    f.write('\t' + f"{word:.6f}")
                f.write('\n')
        return 0

    o, l = convert_features(args.train_input, args.feature_dictionary_in)
    write_o_file(l, o, args.train_out)
    o, l = convert_features(args.validation_input, args.feature_dictionary_in)
    write_o_file(l, o, args.validation_out)
    o, l = convert_features(args.test_input, args.feature_dictionary_in)
    write_o_file(l, o, args.test_out)

    def check_fns(p1, p2):
        d_orig = np.loadtxt(p1, delimiter='\t')
        d_my = np.loadtxt(p2, delimiter='\t')

        for i in range(len(d_orig)):
            for j in range(len(d_my)):
                assert(d_orig[i,j] == d_my[i,j])
    
    # check_fns(args.train_out, "/Users/iskandersergazin/CarngieMellonUniversity/10601/hw4/handout/smalloutput/sample_formatted_train_small.tsv")
    # check_fns(args.test_out, "/Users/iskandersergazin/CarngieMellonUniversity/10601/hw4/handout/smalloutput/sample_formatted_test_small.tsv")
    # check_fns(args.validation_out,"/Users/iskandersergazin/CarngieMellonUniversity/10601/hw4/handout/smalloutput/sample_formatted_val_small.tsv")

    # check_fns(args.train_out,  "/Users/iskandersergazin/CarngieMellonUniversity/10601/hw4/handout/largeoutput/sample_formatted_train_large.tsv")
    # check_fns(args.test_out, "/Users/iskandersergazin/CarngieMellonUniversity/10601/hw4/handout/largeoutput/sample_formatted_test_large.tsv")
    # check_fns(args.validation_out,"/Users/iskandersergazin/CarngieMellonUniversity/10601/hw4/handout/largeoutput/sample_formatted_val_large.tsv")

    print("Hello World")
    



    
