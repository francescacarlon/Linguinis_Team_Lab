import ast
import pandas as pd
import string
import re
import csv
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('wordnet')
nltk.download('punkt')

class Preprocessing:
    def __init__(self):
        self.types_list = []

    # function that cleans the files, tokenizes and gets rid of stopwords
    def clean_files(self, file_path, initial_num):
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

        return lines
    
    # function to remove stopwords from a list of tokens
    def stopwords(self, list):
        stop_words = [
                        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
                        "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her",
                        "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs",
                        "themselves", "what", "which", "who", "whom", "this", "that", "these", "those",
                        "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
                        "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
                        "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",
                        "against", "between", "into", "through", "during", "before", "after", "above",
                        "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
                        "again", "further", "then", "once", "here", "there", "when", "where", "why",
                        "how", "all", "any", "both", "each", "few", "more", "most", "other", "some",
                        "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
                        "s", "t", "can", "will", "just", "don", "should", "now"
                    ]
        
        # create a new list with just the words that are not in the stop_words list
        new_list = []
        for word in list: 
            if word in stop_words:
                continue
            else:
                new_list.append(word)
        return new_list

    # function to tokenize the text in the dataframes    
    def tokenizer(self, df):
        # convert all words in lower case
        df['text'] = df['text'].str.lower()

        # replace "?", "!", "..." with space+string
        df['text'] = df['text'].str.replace('?', ' questionmark ', regex=False)
        df['text'] = df['text'].str.replace('!', ' exclamationmark ', regex=False)

        # replace "\...+" with "ellipsis"
        df['text'] = df['text'].apply(lambda x: re.sub(r'\.\.\.+', ' ellipsismark ', str(x)))

        # replace other punctuation with space
        other_punctuation = r'["#$%&\'()*+,\-./:;<=>@\[\\\]^_`{|}~]'
        df['text'] = df['text'].apply(lambda x: re.sub(other_punctuation, ' ', str(x)))

        # make sure numbers are separated from letters (e.g. '50g' --> '50 g')
        df['text'] = df['text'].apply(lambda x: re.sub(r"([0-9]+)([a-zA-Z]+)", r" \1 \2 ", x))

        # separate tokens
        df['tokens'] = df['text'].apply(lambda x: x.split())

        # remove stopwords
        df["tokens"] = df["tokens"].apply(lambda x: self.stopwords(x))  
        
        return df   
    

    # advanced approaches: lemmatization

    # define a function that performs lemmatization
    # it converts a word into its base form (i.e. the lemma)
    def lemmatization(self, tokens):
        # initialize lemmatizer
        lemmatizer = WordNetLemmatizer()
        # initialize list of lemmas 
        lemmatized_tokens = []
        for word in tokens:
            # specifities of lemmatization. POS is specified (e.g. more precise when comes across 'studied' - 'study'). 
            if word.endswith('s'):
                lemmatized_tokens.append(lemmatizer.lemmatize(word, 'n'))
            elif word.endswith('ed') or word.endswith('ing') or word.endswith('s'):
                lemmatized_tokens.append(lemmatizer.lemmatize(word, 'v'))
            elif word.endswith('ly'):
                lemmatized_tokens.append(word[:-2]) # it does not handle adverbs as well as other POS 
            else:
                lemmatized_tokens.append(word)

        return lemmatized_tokens
    

    # function to process the entire DataFrame: tokenization and lemmatization
    def process_dataframe(self, df):
        tokenized_df = self.tokenizer(df)
        tokenized_df['lemmatized_tokens'] = tokenized_df['tokens'].apply(self.lemmatization)
        return tokenized_df


    # define a function that outputs the vocabulary of the train data. A list of types: ["all", "words", ...]
    def extract_types(self, tokenized_train_df):
        types_list = []
        # for loop to go through all lemmatized_tokens and append the types
        for index, row in tokenized_train_df.iterrows():
            for token in row["lemmatized_tokens"]:
                if token in types_list:
                    continue
                else:
                    types_list.append(token)
        return types_list

