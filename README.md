# Sentiment in Self-Admitted Technical Debt: Analysis and Forecasting


## This project aims to answer followering question:
RQ1: How is the performance of sentiment recognition models based on machine learning algorithms?
To explore the performance of machine learning methods in SATD sentiment recognition, experiments were conducted using an open-source SATD sentiment dataset for model training and testing. The prediction performance of mainstream machine learning algorithms was compared, specifically Naive Bayes, Logistic Regression, Random Forest, SVM, and KNN. These methods are easy to understand and implement, making them accessible for both beginners and professionals. They are highly adaptable, capable of handling both small and large datasets, and work effectively for various types of data, including structured, high-dimensional, and even unstructured data. Additionally, many of these algorithms, such as Random Forest, SVM, and Naive Bayes, offer robust performance in the presence of noise and missing data, providing reliable results across a wide range of real-world applications. The models' performance was evaluated using four metrics: Precision, Recall, F1-score, and AUC (Area Under the Curve).
RQ2: How is sentiment polarity distributed across different types of SATD?
To investigate the differences in sentiment polarity across different types of SATD, we employed the Scott-Knott ESD test[9] method. This method allows for multiple comparisons among groups and identifies those with significant differences in sentiment polarity. We used a manually annotated SATD sentiment dataset, which includes SATD instances extracted from the source code comments of various open-source projects. Each instance is labeled with its corresponding sentiment polarity, including positive, negative, and neutral. By using the Scott-Knott ESD test, we can compare the differences in sentiment polarity distribution among these categories.The results of the Scott-Knott ESD test reveal differences in emotional expression across various types of SATD, providing valuable insights into the mental states of developers when facing different forms of technical debt. These findings can help us better understand the human factors involved in technical debt management and inform the development of more effective SATD management strategies.

Example:
> MyAwesomeProject is a task management tool designed to help users track, categorize, and prioritize tasks efficiently. The app allows users to create, update, and delete tasks, as well as sort them by project or deadline.

## Features
Sentiment Classification: Users can use the model for SATD (Self-Addressed Technical Debt) sentiment classification to automatically identify and classify emotions related to technical debt.
Select the Appropriate Classifier: The project provides models based on five different algorithms, allowing users to choose the most suitable classifier for sentiment analysis.


### Prerequisites
- Python 3.7 or above
- Libraries/Packages:
    pandas: For data manipulation and reading CSV files.
    scikit-learn: For machine learning algorithms, including models like MultinomialNB and SGDClassifier, and tools for text feature extraction (CountVectorizer, TfidfVectorizer), model evaluation, and data splitting.
    NumPy: Required by scikit-learn for numerical operations.
- Tools/Environments:
    IDE or Code Editor:
      Visual Studio Code, PyCharm, or any other Python-friendly IDE to run the project.
    Jupyter Notebook (optional): For running and testing code interactively (especially useful for data science and machine learning projects).
- Operating System:
    The project is cross-platform and should work on any major operating system like Windows, macOS, or Linux.
- Data Files:
    labeled_dataset_0.csv: A CSV file containing labeled data. The dataset should have at least two columns:
        Comment: Text data (input feature).
        Sentiment: Target labels for sentiment analysis.
    labeled_dataset.csv: A CSV file containing Raw Dataset.
    data.csv: The results of the experiment.
- Program Description:
    preprocessing.py:This code is designed to preprocess and clean a dataset containing comments, preparing it for further natural language processing tasks. The main steps include removing URLs, tokenizing the text, performing part-of-speech tagging and lemmatization, and removing stop words. The cleaned data is then saved to a new CSV file.
    model.py:This code is designed to build and evaluate a sentiment analysis model using the Naive Bayes classifier. The process involves loading a preprocessed dataset, vectorizing the text data, training the model, and evaluating its performance.

### Steps to Install
    pip install pandas scikit-learn numpy
    pip install jupyter

### Usage
- Place all files in the same directory as the scripts.
- Run the preprocessing script first:preprocessing.py,Then run the model script:python model.py
