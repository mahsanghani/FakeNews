import prep
import features
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

# naive bayes classifier
nb_pipeline = Pipeline([
    ('NBCV', features.countV),
    ('nb_clf', MultinomialNB())])

nb_pipeline.fit(prep.train_news['text'], prep.train_news['label'])
predicted_nb = nb_pipeline.predict(prep.test_news['text'])

# logistic regression classifier
logR_pipeline = Pipeline([
    ('LogRCV', features.countV),
    ('LogR_clf', LogisticRegression())
])

logR_pipeline.fit(prep.train_news['text'], prep.train_news['label'])
predicted_LogR = logR_pipeline.predict(prep.test_news['text'])

# Linear SVM classifier
svm_pipeline = Pipeline([
    ('svmCV', features.countV),
    ('svm_clf', svm.LinearSVC())
])

svm_pipeline.fit(prep.train_news['text'], prep.train_news['label'])
predicted_svm = svm_pipeline.predict(prep.test_news['text'])

# SVM Stochastic Gradient Descent
sgd_pipeline = Pipeline([
    ('svm2CV', features.countV),
    ('svm2_clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3))
])

sgd_pipeline.fit(prep.train_news['text'], prep.train_news['label'])
predicted_sgd = sgd_pipeline.predict(prep.test_news['text'])

# random forest classifier
random_forest = Pipeline([
    ('rfCV', features.countV),
    ('rf_clf', RandomForestClassifier(n_estimators=200, n_jobs=3))
])

random_forest.fit(prep.train_news['text'], prep.train_news['label'])
predicted_rf = random_forest.predict(prep.test_news['text'])


# K-Fold cross validation
def build_confusion_matrix(classifier):
    k_fold = KFold(n_splits=5)
    scores = []
    confusion = np.array([[0, 0], [0, 0]])

    for train_ind, test_ind in k_fold.split(prep.train_news):
        train_text = prep.train_news.iloc[train_ind]['text']
        train_y = prep.train_news.iloc[train_ind]['label']

        test_text = prep.train_news.iloc[test_ind]['text']
        test_y = prep.train_news.iloc[test_ind]['label']

        classifier.fit(train_text, train_y)
        predictions = classifier.predict(test_text)

        confusion += confusion_matrix(test_y, predictions)
        score = f1_score(test_y, predictions)
        scores.append(score)

    return (print('Total statements classified:', len(prep.train_news)),
            print('Score:', sum(scores) / len(scores)),
            print('score length', len(scores)),
            print('Confusion matrix:'),
            print(confusion))


# K-fold cross validation
build_confusion_matrix(nb_pipeline)
build_confusion_matrix(logR_pipeline)
build_confusion_matrix(svm_pipeline)
build_confusion_matrix(sgd_pipeline)
build_confusion_matrix(random_forest)

# naive-bayes classifier
nb_pipeline_ngram = Pipeline([
    ('nb_tfidf', features.tfidf_ngram),
    ('nb_clf', MultinomialNB())])

nb_pipeline_ngram.fit(prep.train_news['text'], prep.train_news['label'])
predicted_nb_ngram = nb_pipeline_ngram.predict(prep.test_news['text'])

# logistic regression classifier
logR_pipeline_ngram = Pipeline([
    ('LogR_tfidf', features.tfidf_ngram),
    ('LogR_clf', LogisticRegression(penalty="l2", C=1))
])

logR_pipeline_ngram.fit(prep.train_news['text'], prep.train_news['label'])
predicted_LogR_ngram = logR_pipeline_ngram.predict(prep.test_news['text'])

# linear SVM classifier
svm_pipeline_ngram = Pipeline([
    ('svm_tfidf', features.tfidf_ngram),
    ('svm_clf', svm.LinearSVC())
])

svm_pipeline_ngram.fit(prep.train_news['text'], prep.train_news['label'])
predicted_svm_ngram = svm_pipeline_ngram.predict(prep.test_news['text'])

# SGD classifier
sgd_pipeline_ngram = Pipeline([
    ('sgd_tfidf', features.tfidf_ngram),
    ('sgd_clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3))
])

sgd_pipeline_ngram.fit(prep.train_news['text'], prep.train_news['label'])
predicted_sgd_ngram = sgd_pipeline_ngram.predict(prep.test_news['text'])

# random forest classifier
random_forest_ngram = Pipeline([
    ('rf_tfidf', features.tfidf_ngram),
    ('rf_clf', RandomForestClassifier(n_estimators=300, n_jobs=3))
])

random_forest_ngram.fit(prep.train_news['text'], prep.train_news['label'])
predicted_rf_ngram = random_forest_ngram.predict(prep.test_news['text'])

# K-fold cross validation
build_confusion_matrix(nb_pipeline_ngram)
build_confusion_matrix(logR_pipeline_ngram)
build_confusion_matrix(svm_pipeline_ngram)
build_confusion_matrix(sgd_pipeline_ngram)
build_confusion_matrix(random_forest_ngram)

# grid-search hyperparameter optimization
# random forest classifier parameters
parameters = {'rf_tfidf__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)],
              'rf_tfidf__use_idf': (True, False),
              'rf_clf__max_depth': (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
              }

gs_clf = GridSearchCV(random_forest_ngram, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(prep.train_news['text'][:10000], prep.train_news['label'][:10000])

# logistic regression parameters
parameters = {'LogR_tfidf__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)],
              'LogR_tfidf__use_idf': (True, False),
              'LogR_tfidf__smooth_idf': (True, False)
              }

gs_clf = GridSearchCV(logR_pipeline_ngram, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(prep.train_news['text'][:10000], prep.train_news['label'][:10000])

# Linear SVM parameters
parameters = {'svm_tfidf__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)],
              'svm_tfidf__use_idf': (True, False),
              'svm_tfidf__smooth_idf': (True, False),
              'svm_clf__penalty': ('l1', 'l2'),
              }

gs_clf = GridSearchCV(svm_pipeline_ngram, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(prep.train_news['text'][:10000], prep.train_news['label'][:10000])
# GridSearch
random_forest_final = Pipeline([
    ('rf_tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 3), use_idf=True, smooth_idf=True)),
    ('rf_clf', RandomForestClassifier(n_estimators=300, n_jobs=3, max_depth=10))
])

random_forest_final.fit(prep.train_news['text'], prep.train_news['label'])
predicted_rf_final = random_forest_final.predict(prep.test_news['text'])
# np.mean(predicted_rf_final == prep.test_news['label'])
# print(metrics.classification_report(prep.test_news['label'], predicted_rf_final))

logR_pipeline_final = Pipeline([
    ('LogR_tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 5), use_idf=True, smooth_idf=False)),
    ('LogR_clf', LogisticRegression(penalty="l2", C=1))
])

logR_pipeline_final.fit(prep.train_news['text'], prep.train_news['label'])
predicted_LogR_final = logR_pipeline_final.predict(prep.test_news['text'])

# saving best model
model_file = 'final_model.sav'
pickle.dump(logR_pipeline_ngram, open(model_file, 'wb'))


# plot learning curve
def plot_learing_curve(pipeline, title):
    size = 10000
    cv = KFold(size, shuffle=True)

    X = prep.train_news["text"]
    y = prep.train_news["label"]

    pl = pipeline
    pl.fit(X, y)

    train_sizes, train_scores, test_scores = learning_curve(pl, X, y, n_jobs=-1, cv=cv,
                                                            train_sizes=np.linspace(.1, 1.0, 5), verbose=0)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title(title)
    plt.legend(loc="best")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.gca().invert_yaxis()

    # box-like grid
    plt.grid()

    # standard deviation as a range
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")

    # average training and test score lines
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    # shows error from 0 to 1.1
    plt.ylim(-.1, 1.1)
    plt.show()


# plot learning curves for classifiers
plot_learing_curve(logR_pipeline_ngram, "Naive-bayes Classifier")
plot_learing_curve(nb_pipeline_ngram, "LogisticRegression Classifier")
plot_learing_curve(svm_pipeline_ngram, "SVM Classifier")
plot_learing_curve(sgd_pipeline_ngram, "SGD Classifier")
plot_learing_curve(random_forest_ngram, "RandomForest Classifier")


def show_most_informative_features(model, vect, clf, text=None, n=50):
    # get classifier and vectorizer
    vectorizer = model.named_steps[vect]
    classifier = model.named_steps[clf]

    if not hasattr(classifier, 'coef_'):
        raise TypeError(
            "Cannot compute most informative features on {}.".format(
                classifier.__class__.__name__
            )
        )

    if text is not None:
        # text coefficients
        tvec = model.transform([text]).toarray()
    else:
        # or just coefficients
        tvec = classifier.coef_

    # sort names and coefficients together
    coefs = sorted(
        zip(tvec[0], vectorizer.get_feature_names()),
        reverse=True
    )

    # top and bottom coefficient pairs
    topn = zip(coefs[:n], coefs[:-(n + 1):-1])

    # output string
    output = []

    # add predicted value
    if text is not None:
        output.append("\"{}\"".format(text))
        output.append(
            "Classified as: {}".format(model.predict([text]))
        )
        output.append("")

    # most positive and negative features
    for (cp, fnp), (cn, fnn) in topn:
        output.append(
            "{:0.4f}{: >15}    {:0.4f}{: >15}".format(
                cp, fnp, cn, fnn
            )
        )
    print(output)


show_most_informative_features(logR_pipeline_ngram, vect='LogR_tfidf', clf='LogR_clf')
show_most_informative_features(nb_pipeline_ngram, vect='nb_tfidf', clf='nb_clf')
show_most_informative_features(svm_pipeline_ngram, vect='svm_tfidf', clf='svm_clf')
show_most_informative_features(sgd_pipeline_ngram, vect='sgd_tfidf', clf='sgd_clf')
