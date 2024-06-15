import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


class Vectorization:
    def __init__(self):
        pass



    # Function making vector embeddings
    # The function will perform vectorization with the method chosen by user
    def vector_embeddings(my_text, method):

        # EXAMPLE - DELETE LATER!!
        types_list = ['love', 'walk', 'mum', 'icecream', 'house', 'sun', 'summer']
        # my_text = ["I love walking with my mum in the sun sun sun"]

        if method == 'a':  # TF-IDF
            # We use TF-IDF to encode embeddings according to word frequency and its weight (importance) within the sentence
            
            vectorizer = TfidfVectorizer(vocabulary=types_list)
            vectors = vectorizer.fit_transform(my_text)

            # Retrieve the TF-IDF matrix as an array
            tfidf_array = vectors.toarray()

            # Display the results
            feature_names = vectorizer.get_feature_names_out()
            tfidf_df = pd.DataFrame(tfidf_array, columns=feature_names)

            return tfidf_df

        elif method == 'b':  # One-Hot Encoding
            # We use One-Hot Encoding to binary encode vectors embeddings 1-0

            # Create binary vectors for each sentence based on presence of words in types_list
            # We want to represent sentences with one-encoded vectors according to their correspondence in types_list
            # We start by creating a list that will contain the binary_vectors (1, 0) for all sentences

            binary_vectors = []

            # We encode each sentence into a binary vector according to the presence or absence of words from the types_list
            for sentence in my_text:
                binary_vector = []
                sentence_words = sentence.split()
                for word in types_list:
                    if word in sentence_words:
                        binary_vector.append(1)
                    else:
                        binary_vector.append(0)

                binary_vectors.append(binary_vector)
                
            # Display the results
            one_hot_df = pd.DataFrame(binary_vectors, columns=types_list)

            return one_hot_df
    
        else:
            raise ValueError('Invalid choice. Try again.')

        


    def main():
        print('Choose vectorization method:')
        print('a. TF-IDF')
        print('b. One-Hot Encoding')

        method = input("Enter the letter corresponding to your choice: ").strip()


        # TRYOUTS - DELETE LATER
        my_text = ["I love walking with my mum in the sun sun sun", "icecream icecream icecream in summer"]
        # Vectorize data
        vectors = Vectorization.vector_embeddings(my_text, method)
        print("Vectorization completed. \nVectors:", vectors)


if __name__ == "__main__":
    Vectorization.main()


    