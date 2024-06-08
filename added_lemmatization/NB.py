import pandas as pd
import ast
import math
import sys


class NaiveBayes:
    def __init__(self):
        pass

    # for each document, we want to collect its emotion and its lemmatized_tokens in a nested dictionary
    def tokenizer_dict(self, df):
        # create an empty dictionary 
        emotions_tokens_dict = {}

        # for loop to organize info into the dictionary
        for index, row in df.iterrows():
            emotions_tokens_dict[row["ID"]] = {}
            emotions_tokens_dict[row["ID"]]["emotion"] = row["emotion"]
            emotions_tokens_dict[row["ID"]]["lemmatized_tokens"] = row["lemmatized_tokens"]

        return emotions_tokens_dict

    # CREATE GOLD STANDARD DICT
    def get_GS_dict(self, df):
        # create dictionary
        final_dict = {}

        # Iterate over each row in the DataFrame
        for index, row in df.iterrows():
            final_dict[row["ID"]] = row["emotion"]
        
        return final_dict

    # CREATE INDEX_DICTIONARY
    def get_index_lists(self, dict):
        # start a dict where to map emotion and indexes which it appears
        dict_indexes = {}
        # Loop through each item in GS_dict
        for index, emotion in dict.items():
            # Append the index to the correct list in the new dictionary
            if emotion in dict_indexes:
                dict_indexes[emotion].append(index)
            else:
                dict_indexes[emotion] = [index]

        return dict_indexes

    # DEFINING FUNCTION TO CALCULATE PRIOR PROBABILITIES
    # this function computes the likelihood of encountering each class based on how often each class appears in the dataset
    # it generates all prior probabilities and saves them in a dictionary
    # input: dictionary of indexes
    def prior_prob(self, dict_indexes):
        # calculate the prior probability for each class
        # for loop to calculate total number of instances
        total_instances = 0
        for emotion in dict_indexes:
            total_instances += len(dict_indexes[emotion])

        # create a dictionary to append all prior probabilities
        prior_dict = {}
        # for loop to calculate each class prior probability
        for emotion in dict_indexes:
            prior_prob = len(dict_indexes[emotion]) / total_instances
            prior_dict[emotion] = prior_prob

        return prior_dict

    # DENOMINATOR
    # denominator in P(tk|c) --> calculate types in all corpus (vocabulary) + tokens in each class
    # output --> denominator_dict = {"joy": number, "anger": }

    # define a function that outputs the vocabulary of the train data. A list of types: ["all", "words", ...]
    def find_types(self, processed_train_file):
        with open(processed_train_file, encoding='utf-8') as df:
            # read the csv as a DataFrame
            lines = pd.read_csv(df, sep=",")

        # create a list to append all types
        type_list = []

        # for loop to go through all tokens and append the types
        for index, row in lines.iterrows():
            for token in ast.literal_eval(row[-1]):
                if token in type_list:
                    continue
                else:
                    type_list.append(token)

        return type_list

    # DEFINE FUNCTION TO FIND ALL THE TOKENS FOR EVERY CLASS

    # we input the processed train file
    # output a dictionary: emotion_token_dict = {"joy": ["I", "am", "very", "happy",...], "anger":["I", "am", "angry", ...], ... }
    def find_emotion_tokens(self, processed_train_df):
        # create a dictionary to append all emotions and their respective tokens
        emotion_token_dict = {}

        # for loop. For every row in dataframe, we organize the tokens based on which emotion they are labeled with.
        for index, row in processed_train_df.iterrows():
            if row["emotion"] not in emotion_token_dict:
                emotion_token_dict[row["emotion"]] = row["lemmatized_tokens"]
            else:
                emotion_token_dict[row["emotion"]].extend(row["lemmatized_tokens"])

        return emotion_token_dict

    # DEFINE FUNCTION TO FIND ALL DENOMINATORS
    # we input the list of types and the emotion_token_dict
    # it outputs a dictionary with all the classes and their respective denominator for the P(tk|c) calculation
    # output denom_dict = {"joy": 4534534, "sadness": 384834, "anger": 343432, ...}
    def denominator_tk(self, type_list, emotion_token_dict):
        # create dictionary to append all denominators
        denom_dict = {}

        for emotion in emotion_token_dict:
            # denominator is the sum of tokens of each emotion + number of types (for smoothing)
            denominator = len(emotion_token_dict[emotion]) + len(type_list)
            denom_dict[emotion] = denominator

        return denom_dict
    
    # DEFINE FUNCTION TO FIND THE NUMBER OF EACH TOKEN FOR EVERY CLASS
    def count_words_by_emotion(self, type_list, emotion_token_dict):
        # start a dictionary to map types and occurrences of each type in each emotion
        # keys: types
        # values: another dictionary with keys: emotion, values: n. of occurrences of that type in that emotion

        words_by_emotion_dict = {}

        # start with big dictionary with words as keys

        for word in type_list:
            words_by_emotion_dict[word] = {}
            for emotion in emotion_token_dict:
                words_by_emotion_dict[word][emotion] = 1 # start from 1 (smoothing), then fill it in with how many we have

        # for each emotion, get the word count of that specific type
        for emotion, token in emotion_token_dict.items():
            for word in type_list:
                words_by_emotion_dict[word][emotion] += token.count(word)

        return words_by_emotion_dict

    # function that calculates the probabilities for each token to belong to every class
    def token_prob(self, words_by_emotion, denom_dict):
        # create the output dictionary
        token_prob_dict = {}

        # for each type in words_by_emotion, calculate P(tk|c) and append to a nested dictionary
        for token, emotion_count in words_by_emotion.items():
            token_prob_dict[token] = {}
            for emotion, count in emotion_count.items():
                token_prob_dict[token][emotion] = math.log((count + 1)/denom_dict[emotion])

        return token_prob_dict
    
    # function that generates a dictionary with all the model predictions
    def argmax(self, validation_dict, prior_dict, token_prob_dict, denom_dict):
        # initialize dictionary to store final results
        argmax_dict = {}

        # iterate over every instance in validation_dict
        for ID, dictionary in validation_dict.items():
            # set the max probability variable to a very low number (log probabilities will be negative)
            max_probability = -999999  
            # variable for the emotion with the highest probability
            max_emotion = None  

            # for emotion in prior_probabilities_dictionary 
            for emotion, prob in prior_dict.items():
                # calculate the log for the prior probability
                prior_prob = math.log(prior_dict[emotion])
                # set the variable to 0
                sum_token_prob = 0

                # loop to sum the token probabilities 
                for token in dictionary["lemmatized_tokens"]:
                    if token in token_prob_dict:
                        sum_token_prob += token_prob_dict[token][emotion]
                    else:
                        token_prob = math.log(1 / denom_dict[emotion])
                        sum_token_prob += token_prob

                # check which is the emotion with the highest probability
                total_emotion_prob = prior_prob + sum_token_prob
                if total_emotion_prob > max_probability:
                    max_probability = total_emotion_prob
                    max_emotion = emotion

            # add the id as key and the emotion as value in the argmax_dict
            argmax_dict[ID] = max_emotion

        return argmax_dict
