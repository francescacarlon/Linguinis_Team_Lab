import pandas as pd
import string
import re
import csv

# FUNCTION TO ADD IDs TO CSVs.
# The output is the uptdated csv with 3 columns.
# Takes 3 arguments: file path, file name, first number for the indexes column
def add_ID(file_path, filename, initial_num):
  # open csv
  with open(file_path, encoding='utf-8') as df:

      # read the csv as a DataFrame, also adding the labels "emotion" and "text" to the column names
      lines = pd.read_csv(df, sep=',', names=['emotion', 'text'], on_bad_lines = "skip")

      # create a list of desired classes
      emotion_classes = ["joy", "fear", "shame", "disgust", "guilt", "anger", "sadness"]

      # code to fix issue with the original csv. For loop to find if there are any "\n" inside the emotion column.
      for index, row in lines.iterrows():
        if "\n" in row["emotion"]:
          # eliminating the row that is giving problems
          lines = lines.drop(index)
        elif row["emotion"] not in emotion_classes:
          lines = lines.drop(index)

      # adding a new column with the ID numbers for each emotion-text pair
      lines.insert(0, 'ID', range(initial_num, (initial_num+len(lines))))

  # saving the processed file to a csv and saving it to our local computer
  lines.to_csv(filename+"_with_ID.csv", encoding="utf-8", sep=",", index=False)

  return lines


# Generate train file with IDs from 0 to 5326
add_ID("isear-train.csv", "isear_train", 0)

# Generate validation file with IDs from 5327 to 6474
add_ID("isear-val.csv", "isear_val", 5327)

# Generate test file with IDs from 6475 to 7619
add_ID("isear-test.csv", "isear_test", 6475)

# our own dummy evaluation
# We take the isear_val_with_ID.csv and we replace the emotion column with "joy"

# we open the isear validation file with IDs
with open("isear_val_with_ID.csv", "r", encoding="utf-8") as df:
  lines = pd.read_csv(df, sep=",")
  # change the column emotion for "joy"
  lines['emotion'] = "joy"

# save dataframe to a new csv
lines.to_csv("isear_dummy_val_with_ID.csv", encoding="utf-8", sep=",", index=False)


# tokenizer for isear_train_with_ID

pd.set_option('display.max_colwidth', None)

def tokenizer(file_path, file_name):
  # we open the isear file with IDs
  with open(file_path, "r", encoding="utf-8") as x:
    df = pd.read_csv(x, sep=",")

  # convert all words in lower case
  df['text'] = df['text'].str.lower()

  # replace "?", "!", "..." with space+string
  df['text'] = df['text'].str.replace('?', ' question_mark', regex=False)
  df['text'] = df['text'].str.replace('!', ' exclamation_mark', regex=False)

  # replace "\...+" with "ellipsis"
  df['text'] = df['text'].apply(lambda x: re.sub(r'\.\.\.+', 'ellipsis', str(x)))

  # replace other punctuation with space
  other_punctuation = r'["#$%&\'()*+,\-./:;<=>@\[\\\]^_`{|}~]'
  df['text'] = df['text'].apply(lambda x: re.sub(other_punctuation, ' ', str(x)))

  # make sure numbers are separated from letters (e.g. '50g' --> '50 g')
  df['text'] = df['text'].apply(lambda x: re.sub(r"([0-9]+)([a-zA-Z]+)", r"\1 \2", x))

  # separate tokens
  df['tokens'] = df['text'].apply(lambda x: x.split())

  # save the modified DataFrame to a new CSV file
  df.to_csv(file_name+'_tokenized.csv', index=False)


  # print(df['text'][475])
  # print(df['tokens'][475])


# tokenizer for training file
tokenizer("isear_train_with_ID.csv", "isear_train")

# tokenizer for training file
tokenizer("isear_train_with_ID.csv", "isear_train")

# tokenizer for test file
tokenizer("isear_test_with_ID.csv", "isear_test")

# tokenizer for validation file
tokenizer("isear_val_with_ID.csv", "isear_val")