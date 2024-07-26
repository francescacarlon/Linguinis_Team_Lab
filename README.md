# Linguinis_Team_Lab
*Emma Raimundo Schulz, Francesca Carlon*

## Project Description
The project focuses on Emotion Classification, and it aims at classifying texts into classes (i.e. Emotions). 
It is based on the ISEAR dataset.
The goal is to make inference and explore 1) what the best way to evaluate Emotion Classification is and 2) which features can help to get a better classification. 

## Download datasets

### ISEAR (self-reported emotional events) Dataset:

1. **Download** 

   if you want to download it from IMS server:

    CMD: 
    ```
    cd /mount/studenten/team-lab-cl/emotions/isear

    scp user@phoenix:/mount/studenten/team-lab-cl/emotions/isear/isear-test.csv /local/directory/path/
    scp user@phoenix:/mount/studenten/team-lab-cl/emotions/isear/isear-train.csv /local/directory/path/
    scp user@phoenix:/mount/studenten/team-lab-cl/emotions/isear/isear-val.csv /local/directory/path/
    
    ```

   For convenience, you can find isear-test.csv, isear-train.csv, isear-val.csv included in the project folder. 
    
### NRC Word-Emotion Association Lexicon (aka EmoLex)

Downloaded from https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm

Please use the txt file "aNRC-Emotion-Lexicon-Wordlevel-v0.92" included in the project folder. 

## Setup

### Python3
 - version: 3.10.11 or newer

### Virtual Env

1. Install Virtual Env:
```
python3 -m venv .venv
```

2. Activate Virtual Env:
```
# For Linux and MacOS
. .venv/bin/activate 

# For Windows
.venv\Scripts\activate 
```

3. Install dependencies:
```
pip install -r requirements.txt
```


___

- Deactivate Virtual Env:
```
deactivate
```

## Usage

Run main.py file

## Brief Files Descriptions

### main.py

It executes the whole code and performs the following tasks (see following Python files). 
It is possible to change the combination of embedding and model templates to perform in lines 217-219

Example:

```
if __name__ == "__main__":
    embedding_method = config_word2vec_CBOW 
    model_name = config_logistic
```

Note: includes Evaluation. Evaluates model performance and prints relevant metrics.

### DataLoader_processed.py

Loads, cleans, and preprocesses data, including tokenization, lemmatization, and adding sentiment features.

### embeddings_template.py

Converts text data into numerical vectors using the following methods: Tf-Idf, One-hot, CBOW, and Skipgram.

### model_template.py

Selects and trains a model with the following models: SVM, Logistic Regression, Naive Bayes, CNN.

Note: CNN was included but in the end we decided not to make inference with it. 
