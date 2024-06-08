from DataLoader import Preprocessing # type: ignore
from NB import NaiveBayes
from evaluation import Evaluation
from datetime import datetime
import json

def main(): 
    # PREPROCESSING 
    # create an instance of the class Preprocessing to use the functions of the class
    pp = Preprocessing()

    # Generate train file with IDs from 0 to 5326
    train_file = pp.clean_files("isear-train.csv", 0)

    # Generate validation file with IDs from 5327 to 6474
    val_file = pp.clean_files("isear-val.csv", 5327)

    # Generate test file with IDs from 6475 to 7619
    test_file = pp.clean_files("isear-test.csv", 6475)

    # Process the dataframes: it applies tokenization and lemmatization
    processed_train_df = pp.process_dataframe(train_file)
    processed_val_df = pp.process_dataframe(val_file)
    processed_test_df = pp.process_dataframe(test_file)

    # Save the processed data to new CSV files
    processed_train_df.to_csv('processed_train.csv', index=False)
    processed_val_df.to_csv('processed_val.csv', index=False)
    processed_test_df.to_csv('processed_test.csv', index=False)


    """ # generate dataframe with the tokenized train data
    tokenized_train_df = pp.tokenizer(train_file)
    tokenized_val_df = pp.tokenizer(val_file)
    tokenized_test_df = pp.tokenizer(test_file)

    # lemmatization? """

    # generate list of types
    types = pp.extract_types(processed_train_df)

    # NB IMPLEMENTATION
    # create an instance of the class NaiveBayes
    nb = NaiveBayes()

    # make dictionary for tokenized_isear_train_with_ID.csv
    val_dict = nb.tokenizer_dict(processed_val_df)
    test_dict = nb.tokenizer_dict(processed_test_df)

    # create gold standard dict using the training data
    gs_dict = nb.get_GS_dict(processed_train_df)

    # create index_dictionary 
    index_dict = nb.get_index_lists(gs_dict)

    # calculate all prior probabilities and save them to a dictionary
    prior_prob_dict = nb.prior_prob(index_dict)

    # find tokens for every class and organise them into a dictionary
    emotion_token_dict = nb.find_emotion_tokens(processed_train_df)

    # generate a dict with all the denominators to use in P(tk|c)
    denominator_dict = nb.denominator_tk(types, emotion_token_dict)

    # find how many times each type appears in every class
    words_by_emotion_dict = nb.count_words_by_emotion(types, emotion_token_dict)

    # calculate all token probabilities
    token_prob_dict = nb.token_prob(words_by_emotion_dict, denominator_dict)

    # output a dictionary with the model predictions 
    model_prediction = nb.argmax(test_dict, prior_prob_dict, token_prob_dict, denominator_dict)

    # EVALUATION
    # create instance in Evaluation class
    ev = Evaluation()

    # create GS dictionary for testing data
    GS_test_dict = ev.get_GS_dict(test_file)

    # create a GS index list 
    GS_index_dict = ev.get_index_lists(GS_test_dict)

    # create a dictionary of conditions (TP, FP, FN) for every emotion
    conditions_dict = ev.get_conditions(GS_test_dict, model_prediction, GS_index_dict)

    # generate F1 score per class 
    f1_class_dict = ev.get_f1_class(conditions_dict)

    # generate weights for every class
    weight_dict = ev.get_weights(GS_test_dict, GS_index_dict)

    # generate micro and macro f1_score!! 
    final_scores = ev.get_f1_score(f1_class_dict, weight_dict)
    print("Results: ", final_scores)
    
    # save a txt file with the results
    # get current date and time
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    date = datetime.now().strftime("%Y-%m-%d")
    time = datetime.now().strftime("%H:%M:%S")
    print("Date: ", date)
    print("Time: ", time)
    
    # convert date object to string
    str_current_datetime = str(current_date)
    
    # generate a txt file with the results
    file_name = "results_"+str_current_datetime+".txt"
    file = open(file_name, 'w+')

    # add the date and time to the file
    file.write("Date: "+date+"\n")
    file.write("Time: "+time+"\n")

    # add the model results to the file
    file.write(json.dumps(final_scores))

    # print and close file
    print("File created: ", file.name)
    file.close()


if __name__ == "__main__":
    main()
