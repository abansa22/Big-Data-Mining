# Fake News Detection

## 1. Project Introduction

Fake news is a growing concern for individuals and organizations alike. With the advent of
social media and easy access to news on the internet, it is becoming increasingly difficult to
distinguish between real and fake news. In this project, we will classify news articles into real or
fake news. The source of our news articles is the News API and news dataset from Kaggle. We
will use 4 different models to classify the articles: Passive Aggressive Classifier, Multinomial
Naive Bayes, Multilayer Perceptron, and Random Forest Classifier. We will then compare the
results of all models to determine the better-performing one. In this report, we will explain the
development process of a fake news detector. The project involves importing various modules,
creating a method to get news data from the API, getting news sources, getting news using
multiple sources, loading and concatenating the DataFrame, and creating a model for detecting
fake news.

### 1.1 Background

Fake news can be politically driven to give advantages or disadvantages to a political party.
Such news items may contain false and exaggerated claims and, because of certain algorithms,
trap users in a filter bubble. It is creating different issues, from sarcastic articles to fabricated
news and planned government propaganda in some outlets. Fake news and lack of trust in the
media are growing problems with huge complications in our society.

### 1.2 Task Definition

The goal is to detect fake news in order to prevent the spreading of misleading news stories that
come from non-reputable sources. We are seeking to create a model that can predict whether a
given news is fake or real. We can detect fake news using supervised machine learning
methods. In the end, we are expecting to differentiate fake news from real ones.

## 2. Understanding the Data

### 2.1 Data source

The official documentation of the News API claims that there are 3000 authenticated news
sources. I got the news sources using the News API from BBC news. We first got all the
sources from the News API. Then I added the ID of each source to a list. Finally, We truncated
the list to a size of 10 and got news from those sources.
After getting the news sources, we got the news articles from all the news sources and
constructed a new dataset based on those news articles. The News API claims that all of their
news are authenticated. Therefore, we needed another dataset that contains both fake news
and real news to train the model. To create a dataset that consists of fake news and real news,
we loaded the data from a custom file. Finally, it is concatenated to the data fetched from the
News API.

I then split the training and testing data from the dataset. I used 70% of the data for training and
30% for testing.

### 2.2 Data preprocessing

After splitting the training and testing data, I moved to feature selection using the sci-kit-learn
method CountVectorizer(), which is used to convert text to numerical data. This method was
used to remove stop words such as “and,” “the,” and “him,” which are assumed to be
uninformative in representing the content of a text. Therefore they were removed to avoid being
interpreted as a signal for prediction. Similar words can be helpful for prediction in some cases,
such as classifying writing style or personality. I also excluded terms with a document frequency
that is strictly greater than the given threshold (corpus-specific stop words).

### 2.3 Data related process
The following steps are how to develop the dataset:

#### ● Importing the Modules
Import the necessary modules, including NewsApiClient from newsapi, CountVectorizer from
sklearn.feature_extraction.text, PassiveAggressiveClassifier, MultinomialNB, and etc.

● Method to Get News

We created a method to get the news data from the News API. To interact withthe News API, an
API key is required. We used the get_everything() method from the NewsApiClient to get data
and passed various parameters to the method, including sources, domains, etc.

● Get the News Sources

We got all the sources from the News API and added the ID of each source to a list. We then
truncated the list to a size of 10 and got news from those sources.

● Get News Using Multiple Sources

We got news using multiple sources. After getting the news sources, we used the getNews()
method defined earlier to get the news from the API. Next, we created a new data frame using
the news list and added new column headings to the data frame.

● Load and Concat the DataFrame

We loaded and concatenated the data frame using a CSV file containing both fake news and
real news to train the model.

## 2.4 Issues and Difficulties

One of the main issues we encountered was with the News API. We had to use an API key to
interact with the API, which made it difficult to share the code with others. Additionally, we had to
get all the news sources from the API and truncate the list to a size of 10 to get news from those
sources. This meant that we missed some important information in such sources.

## 3. Algorithms

### 3.1 Multilayer Perceptron

A Multilayer Perceptron (MLP) is a type of artificial neural network (ANN) that consists of
multiple layers of nodes, or neurons, arranged in a feedforward manner. It is also known as
Feed-Forward Neural Network (FFNN) or Deep Feed Forward Network (DFFN) in some
literature.

The perceptrons are stacked in multiple layers. In an MLP, data moves from the input to the
output through layers in one (forward) direction. The input layer receives the input data, and the
output layer produces the output. The intermediate layers between the input and output layers
are called hidden layers. Each node in a layer is connected to every node in the adjacent layers
by a weight. The weights are learned during the training process using backpropagation, where
the error between the predicted and actual output is used to adjust the weights.

MLPs are known for their ability to learn complex non-linear relationships between inputs and
outputs, making them useful for a wide range of applications such as image recognition, speech
recognition, and natural language processing. They are widely used in industry and research
due to their effectiveness and flexibility.

The MLP Classifier is used to learn patterns and representations from the raw text data of news
articles in order to classify them as real or fake. It is a supervised learning algorithm, which
means it is trained on labeled data with known article labels (real or fake) to make predictions
on new, unseen data.


### 3.2 Passive Aggressive Classifier

The Passive-Aggressive (PA) algorithm is a type of online machine learning algorithm used for
binary classification tasks. It is a linear classifier that works by updating the model weights
based on the errors made on the training data. The PA algorithm belongs to the family of online
learning algorithms because it can update the model weights on a sample-by-sample basis,
without needing to see the entire training dataset at once. This makes it well-suited for large
datasets that are too big to fit into memory.

The "passive" part of the name refers to the fact that the algorithm makes minimal updates to
the model weights when the training data is classified correctly. The "aggressive" part of the
name refers to the fact that the algorithm makes large updates to the model weights when the
training data is misclassified.

This model is used here is to classify news articles as real or fake. It updates its weights using a
passive-aggressive approach, meaning it makes minimal adjustments to the weights to correct
misclassifications. This makes it efficient for online learning and suitable for classifying news
articles in real time.

It's worth noting that the performance of the Passive Aggressive Classifier may also be
influenced by other factors such as feature engineering, data preprocessing, and model
selection. Experimenting with different hyperparameter values and feature representations or
even trying other classifiers may lead to different results and may be necessary for optimizing
the performance of the model for a specific project.

### 3.3 Random Forest Classifier

Random Forest Classifier is a supervised machine learning algorithm that is used for
classification tasks. It is a type of ensemble learning method that combines the predictions of
multiple decision trees to improve the accuracy of the final model.
The random forest algorithm works by creating a forest of decision trees where each tree is
trained on a random subset of the training data and a random subset of the features. This helps
to reduce overfitting and increases the model's ability to generalize to new data. To make a
prediction, the algorithm uses each decision tree in the forest to predict the class of the input
data point. The final prediction is then based on the majority vote of all the decision trees in the
forest.

Random forest classifiers have several advantages over other machine learning algorithms,
such as being able to handle high-dimensional data and nonlinear relationships between
features and the target variable. They are also robust to noise and missing data and can provide
insight into the importance of different features in the classification task.
Here, the random forest model is another option for classifying news articles as real or fake. It is
an ensemble model that combines multiple decision trees to make predictions. It is known for its
ability to handle high-dimensional data, handle missing values, and avoid overfitting. It can also
capture non-linear relationships between features, which can be beneficial for text classification
tasks.

### 3.4 Multinomial NaiveBayes

Multinomial Naive Bayes is a variant of the Naive Bayes algorithm that is used for text
classification tasks. It is a probabilistic algorithm that works on the assumption that each feature
is independent of all other features, given the class variable. The algorithm works by first
computing the frequency of each word in each document and then using this information to
calculate the probability of each word occurring in each class. These probabilities are then
combined using Bayes' theorem to compute the probability of a given document belonging to
each class. The "multinomial" in the name refers to the fact that the algorithm uses a
multinomial distribution to model the frequency of the words in each document.
Multinomial Naive Bayes is a fast and efficient algorithm that works well for large text datasets. It
is also relatively simple to implement and has been shown to work well in practice for a wide
range of text classification tasks. However, it may not perform as well as more complex
algorithms on tasks where the feature independence assumption is not valid or when the feature
space is very large.

In this project, this model is used to classify news articles as real or fake. Since it is a
probabilistic model based on the Bayes theorem that assumes independence among the
features (words) in the text, it is simple, fast, and performs well on text classification tasks,
making it a popular choice for such applications. It is simple to implement and all you have to do
is calculate probability. This approach works with both continuous and discrete data. It's
straightforward and can be used to forecast real-time applications. It's very scalable and can
handle enormous datasets with ease.

This algorithm's prediction accuracy is lower than that of other probability algorithms. It isn't
appropriate for regression. The Naive Bayes technique can only be used to classify textual input
and cannot be used to estimate numerical values.

## 4. Main Results

The purpose of this analysis is to compare the accuracy of different machine learning models for
differentiating fake news from real ones. Accuracy is a commonly used evaluation metric that
measures the proportion of correct predictions made by a model. In this analysis, I compared
the accuracy of four models: Multilayer Perceptron, Passive Aggressive Classifier, Random
Forest, and Multinomial Naive Bayes.

After training and evaluating the models on a labeled dataset of news articles, the following
testing accuracy values were obtained:

Multilayer Perceptron: 93.09%

Passive Aggressive Classifier: 92.00%

Random Forest: 91.64%

Multinomial Naive Bayes: 90.50%

Based on the accuracy results, the Multilayer Perceptron model achieved the highest accuracy
of 93.09%, making it the best-performing model among the four for detecting fake news. The
Passive Aggressive Classifier also performed well, with an accuracy of 92.00%, followed by the
Random Forest model, with an accuracy of 91.64%. The Multinomial Naive Bayes model had
the lowest accuracy of 90.50% among the four models.

For detecting fake news from real news, the Multilayer Perceptron model demonstrated the
highest accuracy of 93.09%, suggesting it may be a promising choice for this task. However, it's
important to consider other factors, such as model interpretability, explainability, and potential
false positives or false negatives in real-world applications.

We also got the predictions of the models on the testing dataset to check the accuracy of each
model. As can be seen in the slides, MultinomialNB and Random Forest have an accuracy of
95%, while Passive Aggressive and Multilayer Perceptron classifiers have an accuracy of 100%.
This coincides with the accuracies we got earlier, where Multilayer Perceptron and Passive
Aggressive performed the best among all four models. The confusion matrices also show the
above results in a more detailed way.

## 5. Additional Features Added

a. Own Data: Since our dataset was a combination of existing data and data
collected through a News API which we then validated and labelled, we believe
that we fulfilled this component.

b. Implementing more than two algorithms: We have implemented four algorithms -
two of which are popular algorithms for Classification, and the other two are
relatively novel and complex.

## 6. Conclusion and Future Work

In this project, we implemented a news classification system using the data collected by News
API and constructed 4 machine learning models, including Passive Aggressive Classifier,
MultinomialNB, Random Forest Classifier, and MLP Classifier.

The accuracy of the models was evaluated using accuracy score and confusion matrix metrics.

The results showed that all models were able to classify the news articles as real or fake with a
reasonable accuracy and MLP gains the highest performance.

There are several potential areas of improvement and future work that can be explored in this
project:

● More Data Sources: Currently, I have used a limited number of news sources to get
news data. In future work, one can explore using a larger number of news sources to
increase the diversity and coverage of news articles, which may improve the accuracy of
the classification models.

● Model Ensemble: Ensemble techniques such as stacking or blending can be
explored to combine the predictions of multiple models and potentially improve the
overall accuracy and robustness of the classification system.

● Real-time News Classification: Currently, the project gets news data from the News
API for a specific time period. In future work, one can explore real-time news
classification, where news articles are classified as real or fake in real-time as they are
published, using techniques such as streaming data processing and online machine
learning algorithms.

● Semi-supervised learning: The machine learning models used in this project are all
trained in a full-supervised manner. However, in the real-world applications, there are a
large amount of unlabelled fake news. Further exploration of semi-supervised can be
used to optimize the performance fake news detection .

● Utilize the large language model: The project can be further extended to develop a
large fake news detection system with the help of large language model. The system can
also be deployed on a web server or a cloud-based platform to make it trainable along
with the large language model to produce fine-tuned fake news detection embeddings.
