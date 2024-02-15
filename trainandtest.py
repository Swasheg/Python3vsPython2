from os.path import join
import argparse
import pickle
import warnings
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
import numpy as np
from sklearn.model_selection import train_test_split
from keras import backend as K
import utils
from malconv1 import Malconv
from preprocess import preprocess_data
from keras.optimizers import Adam
import seaborn as sns
import os
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Model
import tensorflow as tf
import re
import difflib
from sklearn.model_selection import train_test_split

def get_intermediate_layer_model(model, layer_name):
    return Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

def train(model, x_train, y_train, x_test, y_test, max_len, batch_size, verbose, epochs, save_path):
    ear = EarlyStopping(monitor='val_acc', patience=5)
    mcp = ModelCheckpoint(join(save_path, 'filter128_win5_LR_0.005_altered.h5'),
                          monitor="val_acc",
                          save_best_only=True,
                          save_weights_only=False)

    history = model.fit_generator(
    utils.data_generator(x_train, y_train, max_len, batch_size, shuffle=True),
    steps_per_epoch=len(x_train) // batch_size + 1,
    epochs=epochs,
    verbose=verbose,
    callbacks=[ear, mcp],
    validation_data=utils.data_generator(x_test, y_test, max_len, batch_size),
    validation_steps=len(x_test) // batch_size + 1)
    return history

def get_intermediate_layer_model(model, layer_name):
    return Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

def predict_in_batches(model, data, batch_size, verbose):
    predictions = []
    max_len = model.input.shape[1]

    for i in range(0, len(data), batch_size):
        # Get a batch of data
        batch_data = data[i:i + batch_size]

        # Predict for the current batch
        batch_preds = model.predict(
            utils.data_generator(batch_data, np.zeros(len(batch_data)), max_len, batch_size, shuffle=False),
            steps=len(batch_data) // batch_size + 1,
            verbose=verbose
        )

        # Accumulate predictions
        predictions.extend(batch_preds)

    return predictions

def get_words_around_position(line, position, win_size):
    print(line)
    print(position)
    # Ensure activation position is within bounds
    activation_position = max(win_size, min(len(line) - win_size - 1, position))

    # Extract the word around the activation position
    #start_pos = max(0, activation_position)
    start_pos = position
    end_pos = min(len(line), start_pos + win_size)

    # Extract the word without filtering spaces
    extracted_word = line[start_pos:end_pos]

    return extracted_word

def plot_activations_max_avg1(activations, line, save_folder, idx, preds, activation_list, aggregation='average', win_size=5):
    print(idx)
    sequence_length = len(line)
    fig, ax = plt.subplots(1, 1)

    # Aggregate activations based on the chosen method
    if aggregation == 'max':
        aggregated_activations = np.max(activations[0], axis=-1)
    elif aggregation == 'average':
        aggregated_activations = np.mean(activations[0], axis=-1)
    else:
        raise ValueError("Invalid aggregation method. Use 'max' or 'average'.")

    print(preds[idx])

    # Plot the aggregated activations
    if preds[idx] <= 0.5:
        label = 'python2'
    if preds[idx] > 0.5:
        label = 'python3'
        # Use preds[idx] for the prediction value
    ax.plot(aggregated_activations[:sequence_length])
    ax.set_title(label)
    ax.set_xticks(range(sequence_length))
    ax.set_xticklabels(list(line), rotation=45, fontsize=10)
    os.makedirs(save_folder, exist_ok=True)
    # Save plot as PNG
    plt.savefig(f'{save_folder}/new_line{idx}{aggregation}_new.png')
    plt.close()

    # Get the index of the highest activation
    highest_index = np.argmax(aggregated_activations)
    # Get the word formed from the specified position and window size
    word_around_position = get_words_around_position(line, highest_index, win_size)
    print(word_around_position)
    activation_values = aggregated_activations.tolist()
    activation_list.append({'activations': activation_values, 'word_around_position': word_around_position})
    predict_df.at[idx, 'ExtractedWord'] = word_around_position
    return activation_list

def is_longest_common_substring_gt_3(true_explanation, predicted_explanation,win_size):
    matcher = difflib.SequenceMatcher(None, true_explanation, predicted_explanation)
    match = matcher.find_longest_match(0, len(true_explanation), 0, len(predicted_explanation))
    return match.size > 3

def get_explained_correctly(true_explanation, predicted_explanation,win_size):
    # Check if there is something in 'true explanation' column
    if true_explanation != "":
        return str(is_longest_common_substring_gt_3(true_explanation, predicted_explanation,win_size))
    else:
        return ""
def generate_output_csv(predict_df, output_file_path,win_size):
    # Create a copy of the result_df to avoid modifying the original DataFrame
    output_df = predict_df.copy()

    # Initialize columns with empty values
    output_df['true class'] = ""
    output_df['predicted class'] = ""
    output_df['predicted correctly?'] = ""
    output_df['true explanation'] = ""
    output_df['predicted explanation'] = ""
    output_df['explained correctly?'] = ""
    output_df['accuracy of explanation'] = ""

    # Fill in the values for the specified columns
    output_df['line of code'] = predict_df['FileContent']
    output_df['true class'] = predict_df['Version']
    output_df['predicted class'] = predict_df['predict_score']
    output_df['predicted explanation'] = predict_df['ExtractedWord']

    # Fill in 'predicted correctly?' column based on class prediction
    threshold = 0.5
    output_df['predicted correctly?'] = (abs(predict_df['predict_score'] - predict_df['Version']) < threshold).astype(str)

    patterns = {
        "print(": re.compile(r"print\s*\(\s*(\S+)"),
        "__future__": re.compile(r"(__future__\s+)"),
        "xrange": re.compile(r"(xrange\s*\()"),
        " range": re.compile(r"(\srange\s*\()"),
        'u"': re.compile(r'(u\"\s*(\S+))'),
        "u'": re.compile(r"(u\'\s*(\S+))"),
        "print ": re.compile(r"print\s*\s*(\S+)"),
        "unicode(": re.compile(r"\bunicode\("),
        "__next__(": re.compile(r"\s__next__\s*\("),
        "raw_input(": re.compile(r"\sraw_input\s*\(")
    }

    # Function to extract matched strings for a pattern
    def extract_matches(pattern, text):
        match = re.search(pattern, text)
        return match.group() if match else None

    # Apply the function to each row of the DataFrame
    matched_strings = {key: predict_df['FileContent'].apply(lambda x: extract_matches(pattern, x)) for key, pattern in
                       patterns.items()}

    # Fill in 'true explanation' with the matched strings
    output_df['true explanation'] = ""

    # Fill in 'true explanation' with the matched strings
    output_df['true explanation'] = matched_strings["print("].combine_first(
        matched_strings["__future__"]).combine_first(
        matched_strings["xrange"]).combine_first(
        matched_strings[" range"]).combine_first(
        matched_strings['u"']).combine_first(
        matched_strings['u\'']).combine_first(
        matched_strings["print "]).combine_first(
        matched_strings['unicode(']).combine_first(
        matched_strings['__next__(']).combine_first(
        matched_strings['raw_input('])
    output_df['true explanation']=output_df['true explanation'].fillna('')
    # Fill in 'explained correctly?' column based on the longest common substring condition
    #output_df['explained correctly?'] = output_df.apply(lambda row: str(is_longest_common_substring_gt_3(row['true explanation'], row['predicted explanation'])), axis=1)
    output_df['explained correctly?'] = output_df.apply(lambda row: get_explained_correctly(row['true explanation'], row['predicted explanation'],win_size), axis=1)
   # output_df['explained correctly?'] = output_df.apply(lambda row: str(is_longest_common_substring_gt_3(row['true explanation'], row['predicted explanation'])) if pd.notna(row['true explanation']) else "", axis=1)
    # Calculate accuracy of explanation
    non_empty_true_explanation = output_df['true explanation'] != ""
    non_empty_explained_correctly = output_df[non_empty_true_explanation]['explained correctly?'] == 'True'

    correct_explanation = non_empty_explained_correctly.sum()
    total_non_empty_rows = non_empty_true_explanation.sum()

    # Calculate accuracy percentage
    output_df['accuracy of explanation'] = (correct_explanation / total_non_empty_rows) * 100

    # Reorder the columns as per your requirement
    output_df = output_df[['line of code', 'true class', 'predicted class', 'predicted correctly?',
                           'true explanation', 'predicted explanation', 'explained correctly?',
                           'accuracy of explanation']]

    # Save the DataFrame to a CSV file
    output_df.to_csv(output_file_path, index=False)


def read_raw_text_file(file_path, version):
    with open(file_path, 'r') as file:
        # Read lines from the text file
        lines = file.readlines()

    # Create a DataFrame with the lines and add a 'Version' column
    df = pd.DataFrame({'FileContent': lines})
    df['Version'] = version
    return df

if __name__ == '__main__':


        batch_size = 4
        verbose = 1
        epochs = 100
        limit = 0.
        max_len = 100
        win_size = 5
        val_size = 0.1
        save_path = "/mention/your/save/path/"
        model_path = "/mention/your/model/path.h5"
        result_path = "/mention/your/result/path.csv"
        learning_rate = 0.1

        # limit gpu memory
        if limit > 0:
            utils.limit_gpu_memory(limit)

        model = Malconv(max_len, win_size)
        custom_optimizer = Adam(learning_rate=learning_rate)
        model.compile(loss='binary_crossentropy', optimizer=custom_optimizer, metrics=['acc'])

        # prepare data
        # preprocess is handled in utils.data_generator
        python3paths = '/mention/your/file/path/python2_1.txt'
        python2path = '/mention/your/file/path/python3_1.txt'
        common_path = 'mention/your/file/path/common_1.txt'
        python3_df = read_raw_text_file(python3paths, 1)
        python2_df = read_raw_text_file(python2path, 0)
        #comment the below line to not include the common path
        common_df = read_raw_text_file(common_path, 1)
        # Combine the DataFrames into one big DataFrame
        result_df = pd.concat([python3_df, python2_df, common_df], ignore_index=True)
        #remove comments on the below line to not include the common parts
        #result_df = pd.concat([python3_df, python2_df], ignore_index=True)
        data = result_df['FileContent'].values.tolist()
        print(data)
        data_processed = preprocess_data(data,max_len)
        print(data_processed[0])
        print(result_df['Version'].values)
        x_train, x_test, y_train, y_test = train_test_split(data, result_df['Version'].values,test_size=0.1, random_state=42)
        x_train_preprocessed = preprocess_data(x_train, max_len)
        x_test_preprocessed = preprocess_data(x_test, max_len)
        print('Train on %d data, test on %d data' % (len(x_train), len(x_test)))
        history = train(model, x_train_preprocessed[0], y_train, x_test_preprocessed[0], y_test, max_len, batch_size, verbose, epochs, save_path)
        with open(join(save_path, 'history.pkl'), 'wb') as f:
           pickle.dump(history.history, f)
        predict_df = pd.DataFrame({'FileContent': x_test,'Version':y_test})
        preds = predict_in_batches(model,x_test_preprocessed[0], batch_size, verbose)
        preds = np.vstack(preds)
        predict_df['predict_score'] = preds
        layer_name = 'relu'
        intermediate_model = get_intermediate_layer_model(model, layer_name)
        save_folder = "plots/plots_for_2_classes_with_common_lines/filter128_win5_LR_0.005_altered"
        activation_list = []
        predict_df['ExtractedWord'] = ""
        for idx, (line, data_line) in enumerate(zip(x_test_preprocessed[0], x_test)):
            activations = intermediate_model.predict(np.expand_dims(line, axis=0))
            activation_list = plot_activations_max_avg1(activations, data_line, save_folder, idx, preds, activation_list,aggregation='average',win_size=win_size)
        print(predict_df)
        print('Plots generated successfully.')
        generate_output_csv(predict_df,result_path,win_size)
        print('Results written in', result_path)

