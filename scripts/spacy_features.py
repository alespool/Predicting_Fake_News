from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np
from textstat import textstat
from nrclex import NRCLex
import spacy
from typing import Callable

nlp = spacy.load("en_core_web_sm")

def advanced_feature_extraction(text: str) -> dict:
    """Takes a string as input and returns a dictionary of advanced features.

    Args:
        text (str): The text to process.

    Returns:
        dict: A dictionary with the following keys:
            - 'flesch_reading_ease' (float): The Flesch Reading Ease score.
            - 'gunning_fog' (float): The Gunning Fog score.
            - 'lexical_diversity' (float): The lexical diversity of the text.
            - 'avg_word_length' (float): The average length of the words in the text.
            - 'avg_sentence_length' (float): The average length of the sentences in the text.
            - 'num_entities' (int): The number of named entities in the text.
            - 'noun_to_verb_ratio' (float): The ratio of nouns to verbs in the text.
            - 'positive_emotion_score' (float): The score for positive emotions in the text.
            - 'negative_emotion_score' (float): The score for negative emotions in the text.
    """
    doc = nlp(text)
    emotions = NRCLex(text).affect_frequencies
    
    words = [token.text for token in doc if not token.is_punct]
    num_words = len(words)
    avg_word_length = sum(len(word) for word in words) / num_words if num_words > 0 else 0

    features = {
        'flesch_reading_ease': textstat.flesch_reading_ease(text),
        'gunning_fog': textstat.gunning_fog(text),
        'lexical_diversity': len(set(text.split())) / len(text.split()) if len(text.split()) > 0 else 0,
        'avg_word_length': avg_word_length,
        'avg_sentence_length': sum(len(sent) for sent in doc.sents) / len(list(doc.sents)) if len(list(doc.sents)) > 0 else 0,
        'num_entities': len([ent for ent in doc.ents]),
        'noun_to_verb_ratio': len([token for token in doc if token.pos_ == 'NOUN']) / (len([token for token in doc if token.pos_ == 'VERB']) + 1),
        'positive_emotion_score': emotions.get('positive', 0),
        'negative_emotion_score': emotions.get('negative', 0)
    }

    return features

def parallelize_dataframe(df: pd.DataFrame, func: Callable, num_cores: int) -> pd.DataFrame:
    """
    Uses parallelization to process the dataframe.

    Args:
        df (pd.DataFrame): The dataframe to process.
        func (Callable): The function to apply to each chunk of the dataframe.
        num_cores (int): The number of cores to use for parallelization.

    Returns:
        pd.DataFrame: The processed dataframe.
    """
    df_split = np.array_split(df, num_cores)
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        df = pd.concat(executor.map(func, df_split))
    
    return df

def process_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Process the data we want to get advanced features from in chunks.

    Args:
        chunk (pd.DataFrame): The chunk of data to process.

    Returns:
        pd.DataFrame: The chunk of data with the advanced features column added.
    """
    chunk['advanced_features'] = chunk['text'].apply(advanced_feature_extraction)
    return chunk
