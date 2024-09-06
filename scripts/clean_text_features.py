import re
import string
import matplotlib.pyplot as plt 
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from collections import Counter
from nltk.util import ngrams
from typing import List, Tuple

def plot_wordcloud_and_top_words(
    df: pd.DataFrame, 
    label: int, 
    label_name: str, 
    title_column: str
) -> None:
    """
    Creates a wordcloud for the given dataframe, label, label_name, and title_column.
    The wordcloud is created by joining the title column entries that match the given label
    and removing all non-alphanumeric words and English stopwords.
    Then, the top 10 words in the wordcloud are printed.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the data.
    label : int
        The label to filter the dataframe by.
    label_name : str
        The name of the label
    title_column : str
        The column in the dataframe to use for the wordcloud.
    """
    stop_words = set(stopwords.words('english'))
    titles = ' '.join(df[df['label'] == label][title_column]).lower()
    words = word_tokenize(titles)
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(filtered_words))
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Word Cloud for {title_column} ({label_name}, No Stopwords)')
    plt.axis('off')
    plt.show()
    
    word_counts = Counter(filtered_words)
    top_10_words = word_counts.most_common(10)
    print(f"Top 10 Words for {title_column} ({label_name}):")
    print(top_10_words)

    return filtered_words

def calculate_bigrams_and_trigrams(
    filtered_words_fake: List[str],
    filtered_words_true: List[str]
) -> None:
    """
    Calculates the top 10 bigrams and trigrams from the given filtered words.
    
    Parameters
    ----------
    filtered_words_fake : List[str]
        The filtered words from the fake news data.
    filtered_words_true : List[str]
        The filtered words from the true news data.
    """
    filtered_words: List[str] = filtered_words_fake + filtered_words_true
    
    bigrams: List[Tuple[str, str]] = list(ngrams(filtered_words, 2))
    trigrams: List[Tuple[str, str, str]] = list(ngrams(filtered_words, 3))
    
    bigram_counts: Counter[Tuple[str, str]] = Counter(bigrams)
    trigram_counts: Counter[Tuple[str, str, str]] = Counter(trigrams)
    
    top_10_bigrams: List[Tuple[Tuple[str, str], int]] = bigram_counts.most_common(10)
    top_10_trigrams: List[Tuple[Tuple[str, str, str], int]] = trigram_counts.most_common(10)
    
    print("Top 10 Bigrams:")
    for bigram, count in top_10_bigrams:
        print(f"{bigram}: {count}")
    
    print("Top 10 Trigrams:")
    for trigram, count in top_10_trigrams:
        print(f"{trigram}: {count}")
    
def clean_text(
    text: str,
    remove_repeat_text: bool = True,
    remove_patterns_text: bool = True,
    is_lower: bool = True
) -> str:
    """
    Takes a string as input and returns a cleaned version of the string.

    Args:
        text (str): The input text to be cleaned.
        remove_repeat_text (bool, optional): Whether to remove repeated characters. Defaults to True.
        remove_patterns_text (bool, optional): Whether to remove certain patterns. Defaults to True.
        is_lower (bool, optional): Whether to convert the text to lowercase. Defaults to True.

    Returns:
        str: The cleaned version of the input text.
    """

    if is_lower:
        text = text.lower()

    if remove_repeat_text:
        text = re.sub(r'(.)\1{2,}', r'\1', text)

    if remove_patterns_text:
        # Remove "(Reuters)" and similar patterns
        text = re.sub(r'\b(?:reuters|associated press|ap|bbc|cnn|afp)\b', '', text, flags=re.IGNORECASE)
        # Remove leading location tags like "BEIJING (Reuters) -"
        text = re.sub(r'^\s*[A-Z]+\s*\(\s*[A-Za-z\s]*Reuters\)\s*-\s*', '', text)

    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = text.replace("\n", " ")
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'[0-9]', "", text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Keep only ASCII
    text = re.sub(r'[^A-Za-z\s]', ' ', text)  # Keep only Latin alphabet

    return text
