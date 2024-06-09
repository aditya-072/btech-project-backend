import pandas as pd
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk


def getDataSet():
    return pd.read_csv(
        "medicine_recommendation\datasets\drugLibTrain_raw.tsv", sep="\t"
    )


def get_medicine_recommendation(disease: str = "heart"):

    nltk.download("vader_lexicon")
    data = getDataSet()

    # Filter dataset for a specific disease
    disease_data = data[data["condition"].str.contains(disease, case=False, na=False)]

    # Perform sentiment analysis on reviews
    sid = SentimentIntensityAnalyzer()
    disease_data["sentiment_score"] = disease_data["benefitsReview"].apply(
        lambda x: sid.polarity_scores(x)["compound"]
    )

    # Rank drugs based on sentiment score and other factors
    ranked_drugs = disease_data.sort_values(
        by=["sentiment_score", "effectiveness", "sideEffects"],
        ascending=[False, False, True],
    )

    # Display top recommended drugs
    # print(
    #     ranked_drugs[
    #         [
    #             "urlDrugName",
    #             "sentiment_score",
    #             "effectiveness",
    #             "sideEffects",
    #             "benefitsReview",
    #         ]
    #     ].head(10)
    # )

    return ranked_drugs[
        [
            "urlDrugName",
            "sentiment_score",
            "effectiveness",
            "sideEffects",
            "benefitsReview",
        ]
    ].head(10)
