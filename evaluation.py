import pandas as pd
from collections import defaultdict
import csv


# PREPROCESSING CSV FILE
# Function that takes the csv filepath and the name we want to call it as input 
# It adds a column of IDs to the file and cleans the messy data  
# The output is the uptdated csv with 3 columns
def add_ID(file_path, filename):
  # open csv
  with open(file_path, encoding='utf-8') as df:

      # read the csv as a DataFrame, also adding the labels "emotion" and "text" to the column names
      lines = pd.read_csv(df, sep=',', names=['emotion', 'text'])

      # adding a new column with the ID numbers for each emotion-text pair
      lines.insert(0, 'ID', range(0, len(lines)))

      # code to fix issue with the original csv. For loop to find if there are any "\n" inside the emotion column.
      for index, row in lines.iterrows():
        if "\n" in row["emotion"]:
          # eliminating the row that is giving problems
          lines = lines.drop(index)

  # saving the processed file to a csv and saving it to our local computer
  lines.to_csv(filename+"_with_ID.csv", encoding="utf-8", sep=",")

  return lines

# GOLD STANDARD FULL DICT
# Takes the csv data_with_IDs as input
# Converts the csv to a dictionary of indexes and their corresponding emotion
def get_GS_dict(file_path):
  # open the file
  with open(file_path, encoding='utf-8') as df:

      # read the csv as a DataFrame, also adding the labels "emotion" and "text" to the column names
      lines = pd.read_csv(df, sep=',')

      # create dictionary
      final_dict = {}

      # Iterate over each row in the DataFrame
      for index, row in data.iterrows():
        final_dict[row["ID"]] = row["emotion"]
      return final_dict
  

# GOLD STANDARD INDEX LIST
# Takes GS_dictionary as input
# Outputs a new dictionary with every emotion as key and their respective indexes in the GS
def get_index_lists(dict):
  # start a dict where to map emotion and indexes which it appears
  dict_indexes = {}
  # Loop through each item in GS_dict
  for key, value in dict.items():
    # Append the index to the correct list in the new dictionary
    if value not in dict_indexes:
        dict_indexes[value] = [key]
    else:
      dict_indexes[value].append(key)

  return dict_indexes


# GET A DICTIONARY OF CONDITIONS
# input: GS_dictionary, model_predictions_dictionary (which will be our model output) and the dictionary of indexes
# outputs a dictionary with every emotion as key. Their values are another dictionary with their respective TP, FP and FN. 
def get_conditions(gold_standard = dict, model_predictions = dict, index_list = dict):
  # create a new final dictionary to store all conditions
  all_conditions = {}
  # for loop to add the every emotion key in the all_conditions dictionary
  for emotion in index_list:
    # add emotion as key and a dictionary with the conditions inside
    all_conditions[emotion] = {"TP": 0, "FP": 0, "FN": 0}

  # for every emotion, we need to calculate its TP, FP and TN
  # for emotion in index list
  for emotion in index_list:
    # for index number in GS dictionary:
    for index in gold_standard:
      if gold_standard[index] == emotion and model_predictions[index] == emotion:
         # sum 1 to the key "TP" in the corresponding emotion, in the all_conditions dictionary
         all_conditions[gold_standard[index]]["TP"] += 1

      elif gold_standard[index] == emotion and model_predictions[index] != emotion:
        # sum 1 to the key "FN" in the corresponding emotion, in the all_conditions dictionary
         all_conditions[gold_standard[index]]["FN"] += 1

      elif gold_standard[index] != emotion and model_predictions[index] == emotion:
        # sum 1 to the key "FP" in the corresponding emotion, in the all_conditions dictionary
         all_conditions[gold_standard[index]]["FP"] += 1

  return all_conditions


# GENERATE F1 SCORE PER CLASS
# function that accepts a dict of all the conditions as input
# output is a dict with with all the F1 scores
def get_f1_class(conditions = dict):

  # create a dictionary to store all f1 scores
  f1_dict = {}
  for emotion in conditions:
    # calculate precision (how many instances the model predicted correctly)
    precision = conditions[emotion]["TP"] / (conditions[emotion]["TP"] + conditions[emotion]["FP"])
    print(emotion, " precision: ", precision)
    # calculate recall (how many of the true labels in the Gold Standard were correctly classified but the model)
    recall = conditions[emotion]["TP"] / (conditions[emotion]["TP"] + conditions[emotion]["FN"])
    print(emotion, " recall: ", recall)
    # calculate f1 score
    f1_score = (2 * precision * recall) / (precision + recall)
    print(emotion, " F1 score: ", f1_score)
    f1_dict[emotion] = f1_score

  return f1_dict


# GET WEIGHTS
# function that takes the GS_dict and the list of indexes dict
# outputs a dictionary with every emotion's weight
def get_weights(gold_standard = dict, index_list = dict):
  # create a dictionary to store the weights
  weight_dict = {}
  # find weights for each emotion
  for emotion in index_list:
    weight = len(index_list[emotion]) / len(gold_standard)
    weight_dict[emotion] = weight
  return weight_dict


# MICRO AND MACRO F1_SCORE
# function that takes the dictionary with the f1 scores as input
# outputs dictionary with macro and micro f1 score
def get_f1_score(f1_dict, weights):
  # create dictionary for results
  results = {}
  # define variable for summing the f1 scores
  total_f1 = 0
  # macro f1 score (average between all per-class f1 scores)
  for emotion in f1_dict:
    total_f1 += f1_dict[emotion]
  macro = total_f1 / len(f1_dict)
  results["macro F1 score"] = macro

  # micro f1 score (weighted average)
  micro = 0
  for emotion in f1_dict:
    weighed_emotion = f1_dict[emotion]*emo_weights[emotion]
    micro += weighed_emotion
    results["micro F1 score"] = micro

  return results

