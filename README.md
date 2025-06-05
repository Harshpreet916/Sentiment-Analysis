# Sentiment-Analysis

---

# Sentiment Analysis of Food Delivery Reviews

## Project Overview

This project focuses on analyzing customer reviews from Zomato and Swiggy to classify sentiments (e.g., positive, negative) using Natural Language Processing (NLP) and machine learning techniques. It includes data preprocessing, exploratory data analysis (EDA), visualization, and model building.

---

## Features

* **NLP Pipeline**: Text cleaning, stopword removal, stemming, and vectorization using CountVectorizer and TfidfVectorizer.
* **EDA and Visualization**: Insights into the data using Seaborn and Matplotlib, including word clouds and count plots.
* **Machine Learning Models**: Sentiment classification using Multinomial Naive Bayes and evaluation with metrics like accuracy and classification reports.

---

## Project Architecture

<a>
  <img align="left" align="centre" alt="Project Architecture" width="400px" src="sentinment analysis architecture.png" />
</a>



</br>

1. **Data Collection and Preprocessing**

   * Datasets: Customer reviews from Zomato and Swiggy platforms.
   * Preprocessing:

     * Conversion to lowercase.
     * Removal of non-alphabetic characters.
     * Stopword removal using NLTK.
     * Stemming using SnowballStemmer.

2. **Exploratory Data Analysis (EDA)**

   * Word frequency analysis and rating distribution.
   * Word cloud generation for visual insights.
   * Count plots for rating distribution.

3. **Model Building and Evaluation**

   * **Vectorization**: Bag-of-Words representation using CountVectorizer and TfidfVectorizer.
   * **Models**:

     * Multinomial Naive Bayes.
     * Logistic Regression (optional extension).
   * **Evaluation Metrics**: Accuracy, classification reports, and confusion matrices.

4. **Visualization**

   * Word clouds for top words in reviews.
   * Count plots for distribution of ratings.

---

## Tools and Technologies

* **Programming Language**: Python
* **Libraries**: Pandas, NumPy, NLTK, Scikit-learn, Matplotlib, Seaborn, WordCloud
* **Other**: GitHub for version control

---

## Installation and Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/sentiment-analysis.git
   cd sentiment-analysis
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook or script:

   ```bash
   python main.py
   ```

---

## How to Use

1. Place your dataset files (`zomato.csv` and `swiggy.csv`) in the `data` directory.
2. Run the preprocessing script to clean and prepare the data.
3. Train the model and visualize the results using the provided scripts.

---

## Results

* Models achieved a sentiment classification accuracy of \[insert value].
* Key insights were visualized through EDA and word clouds, highlighting the trends in customer feedback.

---

## Future Enhancements

* Support for additional food delivery platforms.
* Incorporation of advanced NLP models like BERT for improved accuracy.
* Deployment as a web application for real-time sentiment analysis.

---

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.


