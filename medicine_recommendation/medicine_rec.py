import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")


def preprocess(sentence):

    tokens = word_tokenize(sentence.lower())

    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    return " ".join(lemmatized_tokens)


def sentence_similarity(sentence1, sentence2):

    preprocessed_sentence1 = preprocess(sentence1)
    preprocessed_sentence2 = preprocess(sentence2)

    tfidf_vectorizer = TfidfVectorizer()

    tfidf_matrix = tfidf_vectorizer.fit_transform(
        [preprocessed_sentence1, preprocessed_sentence2]
    )

    similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

    return similarity_score


def recommend(sen: str, disease: str):
    data = pd.read_csv(
        "medicine_recommendation\\datasets\\drugLibTrain_raw.tsv", sep="\t"
    )

    disease_data = data[data["condition"].str.contains(disease, case=False, na=False)]

    review = disease_data["benefitsReview"]
    rev_idx = review.keys()
    list = []
    for i in range(0, len(rev_idx)):
        if review[rev_idx[i]] is not None:

            similarity_score = sentence_similarity(review[rev_idx[i]], sen)
            if similarity_score > 0:
                print(disease_data.loc[rev_idx[i]]["urlDrugName"], similarity_score)
                list.insert(0, disease_data.loc[rev_idx[i]]["urlDrugName"])
    print(f"the medicines you need are:{list}")
    list_ = []
    for li in list:
        if li not in list_:
            list_.append(li)
    return list_
