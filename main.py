"""
Takes text from a tabular file and outputs sentiment scores and/or keyword
extraction results for each record of text provided.
"""
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import yake

INPUT_FILEPATH = './data/sightings_with-coords.csv'
OUTPUT_FILEPATH = './data/sightings_with-coords-keywords.xlsx'


def add_sentiment(dataframe: pd.DataFrame,
                  text_column: str) -> pd.DataFrame:
    """
    Adds sentiment analysis data to a dataframe
    :param dataframe: Dataframe to change
    :param text_column: Label of the column in the df to analyze
    :return: Pandas DataFrame
    """
    sid = SentimentIntensityAnalyzer()

    scores = dataframe[text_column].apply(
        lambda x: pd.Series(sid.polarity_scores(str(x)))
    )

    scores.rename(axis=1, inplace=True, mapper={
        'neg': 'sentiment_negative',
        'neu': 'sentiment_neutral',
        'pos': 'sentiment_positive',
        'compound': 'sentiment_compound'
    })

    return pd.concat([dataframe, scores], axis=1)


def add_keywords(dataframe: pd.DataFrame,
                 text_column: str,
                 num_keywords: int) -> pd.DataFrame:
    """
    Adds keyword data to a dataframe
    :param dataframe: Dataframe to change
    :param text_column: Label of the column in the df to analyze
    :param num_keywords: Maximum number of keywords desired
    :return:
    """
    kw_extractor = yake.KeywordExtractor(lan='en', n=2, dedupLim=0.1,
                                         top=num_keywords)

    # creates an n-column df with the keywords; excludes yake's h-value
    keywords = dataframe[text_column].apply(
        lambda x: pd.Series(
            [keyword[0] for keyword in kw_extractor.extract_keywords(str(x))])
    )

    column_name_mapper = {}
    for i in range(num_keywords):
        column_name_mapper[i] = f'keyword_{i+1}'

    keywords.rename(axis=1, inplace=True, mapper=column_name_mapper)

    return pd.concat([dataframe, keywords], axis=1)


if __name__ == '__main__':
    df = pd.read_csv(INPUT_FILEPATH)

    # SET THESE
    column_to_analyze = 'Summary'
    num_keywords_desired = 3

    # comment out one of the functions below if you don't need it
    df = add_sentiment(df, column_to_analyze)
    df = add_keywords(df, column_to_analyze, num_keywords_desired)

    df.to_excel(OUTPUT_FILEPATH)