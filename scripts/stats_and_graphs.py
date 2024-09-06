import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

def plot_text_statistics(fake_df, true_df):
    """
    Plots histograms and boxplots for word count, sentence count, and token count
    for both fake and true datasets.
    
    Parameters:
    fake_df (DataFrame): DataFrame containing the fake news data with text column.
    true_df (DataFrame): DataFrame containing the true news data with text column.
    """

    if 'token_count' not in fake_df.columns or 'token_count' not in true_df.columns:
        raise ValueError("The DataFrames must have 'token_count' column. Please compute it before plotting.")

    fig, axs = plt.subplots(3, 2, figsize=(12, 15))

    axs[0, 0].hist(fake_df['word_count'], bins=20, alpha=0.7, label='Fake Data')
    axs[0, 0].hist(true_df['word_count'], bins=20, alpha=0.7, label='True Data')
    axs[0, 0].set_xlabel('Word Count')
    axs[0, 0].set_ylabel('Frequency')
    axs[0, 0].set_title('Word Count Distribution')
    axs[0, 0].legend()

    axs[0, 1].boxplot([fake_df['word_count'], true_df['word_count']], labels=['Fake Data', 'True Data'])
    axs[0, 1].set_xlabel('Data Type')
    axs[0, 1].set_ylabel('Word Count')
    axs[0, 1].set_title('Word Count Boxplot')

    axs[1, 0].hist(fake_df['sentence_count'], bins=20, alpha=0.7, label='Fake Data')
    axs[1, 0].hist(true_df['sentence_count'], bins=20, alpha=0.7, label='True Data')
    axs[1, 0].set_xlabel('Sentence Count')
    axs[1, 0].set_ylabel('Frequency')
    axs[1, 0].set_title('Sentence Count Distribution')
    axs[1, 0].legend()

    axs[1, 1].boxplot([fake_df['sentence_count'], true_df['sentence_count']], labels=['Fake Data', 'True Data'])
    axs[1, 1].set_xlabel('Data Type')
    axs[1, 1].set_ylabel('Sentence Count')
    axs[1, 1].set_title('Sentence Count Boxplot')

    axs[2, 0].hist(fake_df['token_count'], bins=20, alpha=0.7, label='Fake Data')
    axs[2, 0].hist(true_df['token_count'], bins=20, alpha=0.7, label='True Data')
    axs[2, 0].set_xlabel('Token Count')
    axs[2, 0].set_ylabel('Frequency')
    axs[2, 0].set_title('Token Count Distribution')
    axs[2, 0].legend()

    axs[2, 1].boxplot([fake_df['token_count'], true_df['token_count']], labels=['Fake Data', 'True Data'])
    axs[2, 1].set_xlabel('Data Type')
    axs[2, 1].set_ylabel('Token Count')
    axs[2, 1].set_title('Token Count Boxplot')

    plt.tight_layout()
    plt.show()

def compare_columns(fake_df, true_df, column_name):
    """
    Compares the given column between the fake and true news datasets using an independent samples t-test.

    Parameters:
    fake_df (DataFrame): The DataFrame containing the fake news data.
    true_df (DataFrame): The DataFrame containing the true news data.
    column_name (str): The name of the column to compare.

    Returns:
    tuple: A tuple containing the t-statistic and p-value of the t-test.

    Raises:
    ValueError: If the column is not present in both DataFrames.
    """
    if column_name not in fake_df.columns or column_name not in true_df.columns:
        raise ValueError(f"The column '{column_name}' must be present in both DataFrames.")
    
    t_stat, p_val = ttest_ind(fake_df[column_name], true_df[column_name])
    
    print(f"t-statistic: {t_stat:.4f}, p-value: {p_val:.4f}")
    
    if p_val < 0.05:
        print(f"There is a significant difference in '{column_name}' between fake and true news.")
    else:
        print(f"There is no significant difference in '{column_name}' between fake and true news.")
    
    return t_stat, p_val
