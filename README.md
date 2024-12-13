### PREDICTING FAKE NEWS

---

## Project Overview

In this project, we are tasked with understanding the data differences between true and fake news, and create models to predict when an article will be classified as fake news. This repository contains two Jupyter notebooks designed for natural language processing (NLP) tasks related to fake and real news classification. The project leverages deep learning techniques and various data preprocessing methods to differentiate between fake and real news articles.

### Notebooks Included

1. **Deep Learning Model Notebook**
   - **Filename**: `deep_learning_model.ipynb`
   - **Purpose**: This notebook is focused on developing and training a deep learning model for classifying news articles as fake or real. It includes the following steps:
     - Data loading and preprocessing
     - Text processing and feature extraction
     - Model building using Pycharm deep learning frameworks and pre-existent RoBERTa model
     - Model training and evaluation
     - Visualization of results

2. **Fake and Real Data Analysis Notebook**
   - **Filename**: `fake_real_data.ipynb`
   - **Purpose**: This notebook is centered on data exploration and preprocessing as well as modelling. It includes:
     - Loading the fake and real news datasets
     - Exploratory data analysis (EDA)
     - Data cleaning and preprocessing
     - Visualization of data distributions and trends
     - Preparation of data for use in machine learning models
     - Fine-tuning and training of different classical machine learning models for predicting fake news

### Getting Started

To use these notebooks, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Install required libraries**:
   Ensure you have Python installed, and then install the necessary packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the datasets**:
   Place the `Fake.csv` and `True.csv` datasets in the `data/` directory. The notebooks assume this directory structure.

4. **Run the Notebooks**:
   Open the notebooks using Jupyter Lab or Jupyter Notebook and run the cells sequentially to replicate the analysis.

### Notable Visualizations

Here are some key visualizations included in the notebooks:

- **Data Distribution Plots**: Shows the distribution of word counts, sentence lengths, and other features for both fake and real news articles.
- **Confusion Matrix**: Displays the performance of the deep learning model on the test set, illustrating the number of true positives, true negatives, false positives, and false negatives.

![RoBERTa Confusion Matrix](./images/confusion_matrix_roberta.png)

![RoBERTa ROC AUC Score](./images/auc_roberta.png)

### File Structure

```
├── data/
│   ├── Fake.csv
│   ├── True.csv
├── deep_learning_model.ipynb
├── fake_real_data.ipynb
├── requirements.txt
└── README.md
```

### Additional details

In addition to the plots and results shown above, in this project we are also using the SHAP library to understand more in detail the results of our model, and a brief research on the effects of using such a model with its confidence levels is pursued. 

![LogReg SHAP](./images/SHAP_log_reg.png)

![LogReg Confidence](./images/confidence_levels_log_reg.png)

To top it off, we reach a final average confidence score divided by correct and wrong predictions:

    - Average Confidence in Correct Predictions: 0.9534557995194625
    - Average Confidence in Wrong Predictions: 0.7006026043397466


### Contributing

Contributions are welcome! Feel free to submit a pull request or open an issue to discuss any improvements or bugs.

### License

This project is licensed under the MIT License.
