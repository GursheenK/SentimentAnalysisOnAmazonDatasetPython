
# Sentiment Analysis On Amazon Dataset

Sentiment analysis (or opinion mining) is a natural language processing (NLP) technique used to determine whether data is positive, negative or neutral. Sentiment analysis is often performed on textual data to help businesses monitor brand and product sentiment in customer feedback, and understand customer needs.

## Steps

- Import the required packages.
- Remove the null values using the dropna function.
- Sample the dataset.
- Calculating the sentiments based on the conditionRating < 3 — Negative(0)
Rating >=3 — Positive(1)
- Split the dataset into training and testing sets.
- Vectorize the data using CountVectorizer.
- Train the SVM Classifier model.
- Calculate the score of the model.
- Predict the result for the testing dataset.
- Check the output.


## Libraries Used

### sklearn
scikit-learn is an open-source Python library that implements a range of machine learning, pre-processing, cross-validation, and visualization algorithms using a unified interface.

### pandas
Pandas is a Python library used for working with data sets. It has functions for analyzing, cleaning, exploring, and manipulating data.




## Installation

Using pip

```bash
   pip install sklearn
```
    
## Authors

[@Gursheen Kaur Anand](https://github.com/GursheenK/)


