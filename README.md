# Linguinis_Team_Lab
*Emma Raimundo Schulz, Francesca Carlon*

## Project Description
The project focuses on Emotion Classification, and it aims at classifying texts into classes (i.e. Emotions). 
It is based on the ISEAR dataset.

## Download datasets

### ISEAR (self-reported emotional events) Dataset:

1. **Download** 

   from IMS server:

    CMD: 
    ```
    cd /mount/studenten/team-lab-cl/emotions/isear

    scp user@phoenix:/mount/studenten/team-lab-cl/emotions/isear/isear-test.csv /local/directory/path/
    
    ```


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

## How To Run

