# FakeNews
Fake News Classification

### prep.py
Each and every pre-processing feature required to handle all incoming texts and documents is contained in this file. Before beginning any preprocessing, such as tokenizing or stemming, I first read the train, test, and validation data files. The distribution of answer variables and data quality checks, such as the use of null or missing values, are a few examples of exploratory data analysis.

### features.py
I used feature extraction and selection techniques from the sci-kit learn python libraries in this file. I've used techniques like basic bag-of-words and n-grams for feature selection, followed by term frequency techniques like tf-tdf weighting. Although POS tagging and word2vec have not yet been employed in the project, I have also used them to extract the features.

### classifier.py
Here, I have created every classifier for detecting bogus news. Various classifiers are given the extracted characteristics. I utilised classifiers from Sklearn's Naive-Bayes, Logistic Regression, Linear SVM, Stochastic Gradient Descent, and Random Forest. All the classifiers utilised each of the retrieved characteristics. I compared the f1 score and looked at the confusion matrix after fitting the model. Two of the best-performing models were chosen as candidate models for the categorization of false news after fitting all the classifiers. 

On the previously mentioend model candidates, I applied GridSearchCV techniques for parameter tweaking, and I selected the best performing parameters for these classifiers. The final model chosen was utilised to detect bogus news with a probability of truth.  Furthermore, in order to determine which terms are most prevalent and significant in each of the classes, I have also retrieved the top 50 features from our term-frequency tfidf vectorizer. In order to determine how well the training and test sets work as I increase the quantity of data in our classifiers, I have also employed Precision-Recall and learning curves.

### predict.py
Finally, Logistic Regression was determined to be the highest performing classifier, and it was saved on disc as a pickle file with the name model.pkl in a compressed format before being published to a github repository. This model will be cloned, copied to a local directory, and utilised by the predict.py file to identify phoney news. The user input is a news story, and the model is then utilised to produce the final classification output, which is displayed to the user along with the prediction probability.

### data/
train.csv
test.csv
labels.csv
glove6B50d.txt.zip

### requirements.txt
to be installed with the help of the following command:

```pip3 install -r /path/to/requirements.txt```