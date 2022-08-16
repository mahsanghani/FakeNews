import prep
import FeatureSelection
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
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

# naive bayes classifier
nb_pipeline = Pipeline([
    ('NBCV', FeatureSelection.countV),
    ('nb_clf', MultinomialNB())])

nb_pipeline.fit(prep.train_news['Statement'], prep.train_news['Label'])
predicted_nb = nb_pipeline.predict(prep.test_news['Statement'])
np.mean(predicted_nb == prep.test_news['Label'])

# logistic regression classifier
logR_pipeline = Pipeline([
    ('LogRCV', FeatureSelection.countV),
    ('LogR_clf', LogisticRegression())
])

logR_pipeline.fit(prep.train_news['Statement'], prep.train_news['Label'])
predicted_LogR = logR_pipeline.predict(prep.test_news['Statement'])
np.mean(predicted_LogR == prep.test_news['Label'])

# Linear SVM classifier
svm_pipeline = Pipeline([
    ('svmCV', FeatureSelection.countV),
    ('svm_clf', svm.LinearSVC())
])

svm_pipeline.fit(prep.train_news['Statement'], prep.train_news['Label'])
predicted_svm = svm_pipeline.predict(prep.test_news['Statement'])
np.mean(predicted_svm == prep.test_news['Label'])

# SVM Stochastic Gradient Descent
sgd_pipeline = Pipeline([
    ('svm2CV', FeatureSelection.countV),
    ('svm2_clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5))
])

sgd_pipeline.fit(prep.train_news['Statement'], prep.train_news['Label'])
predicted_sgd = sgd_pipeline.predict(prep.test_news['Statement'])
np.mean(predicted_sgd == prep.test_news['Label'])

# random forest classifier
random_forest = Pipeline([
    ('rfCV', FeatureSelection.countV),
    ('rf_clf', RandomForestClassifier(n_estimators=200, n_jobs=3))
])

random_forest.fit(prep.train_news['Statement'], prep.train_news['Label'])
predicted_rf = random_forest.predict(prep.test_news['Statement'])
np.mean(predicted_rf == prep.test_news['Label'])

# K-Fold cross validation
def build_confusion_matrix(classifier):
    k_fold = KFold(n_splits=5)
    scores = []
    confusion = np.array([[0, 0], [0, 0]])

    for train_ind, test_ind in k_fold.split(prep.train_news):
        train_text = prep.train_news.iloc[train_ind]['Statement']
        train_y = prep.train_news.iloc[train_ind]['Label']

        test_text = prep.train_news.iloc[test_ind]['Statement']
        test_y = prep.train_news.iloc[test_ind]['Label']

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
    ('nb_tfidf', FeatureSelection.tfidf_ngram),
    ('nb_clf', MultinomialNB())])

nb_pipeline_ngram.fit(prep.train_news['Statement'], prep.train_news['Label'])
predicted_nb_ngram = nb_pipeline_ngram.predict(prep.test_news['Statement'])
np.mean(predicted_nb_ngram == prep.test_news['Label'])

# logistic regression classifier
logR_pipeline_ngram = Pipeline([
    ('LogR_tfidf', FeatureSelection.tfidf_ngram),
    ('LogR_clf', LogisticRegression(penalty="l2", C=1))
])

logR_pipeline_ngram.fit(prep.train_news['Statement'], prep.train_news['Label'])
predicted_LogR_ngram = logR_pipeline_ngram.predict(prep.test_news['Statement'])
np.mean(predicted_LogR_ngram == prep.test_news['Label'])

# linear SVM classifier
svm_pipeline_ngram = Pipeline([
    ('svm_tfidf', FeatureSelection.tfidf_ngram),
    ('svm_clf', svm.LinearSVC())
])

svm_pipeline_ngram.fit(prep.train_news['Statement'], prep.train_news['Label'])
predicted_svm_ngram = svm_pipeline_ngram.predict(prep.test_news['Statement'])
np.mean(predicted_svm_ngram == prep.test_news['Label'])

# SGD classifier
sgd_pipeline_ngram = Pipeline([
    ('sgd_tfidf', FeatureSelection.tfidf_ngram),
    ('sgd_clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5))
])

sgd_pipeline_ngram.fit(prep.train_news['Statement'], prep.train_news['Label'])
predicted_sgd_ngram = sgd_pipeline_ngram.predict(prep.test_news['Statement'])
np.mean(predicted_sgd_ngram == prep.test_news['Label'])

# random forest classifier
random_forest_ngram = Pipeline([
    ('rf_tfidf', FeatureSelection.tfidf_ngram),
    ('rf_clf', RandomForestClassifier(n_estimators=300, n_jobs=3))
])

random_forest_ngram.fit(prep.train_news['Statement'], prep.train_news['Label'])
predicted_rf_ngram = random_forest_ngram.predict(prep.test_news['Statement'])
np.mean(predicted_rf_ngram == prep.test_news['Label'])

# K-fold cross validation
build_confusion_matrix(nb_pipeline_ngram)
build_confusion_matrix(logR_pipeline_ngram)
build_confusion_matrix(svm_pipeline_ngram)
build_confusion_matrix(sgd_pipeline_ngram)
build_confusion_matrix(random_forest_ngram)

print(classification_report(prep.test_news['Label'], predicted_nb_ngram))
print(classification_report(prep.test_news['Label'], predicted_LogR_ngram))
print(classification_report(prep.test_news['Label'], predicted_svm_ngram))
print(classification_report(prep.test_news['Label'], predicted_sgd_ngram))
print(classification_report(prep.test_news['Label'], predicted_rf_ngram))

prep.test_news['Label'].shape

# grid-search hyperparameter optimization
# random forest classifier parameters
parameters = {'rf_tfidf__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)],
              'rf_tfidf__use_idf': (True, False),
              'rf_clf__max_depth': (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
              }

gs_clf = GridSearchCV(random_forest_ngram, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(prep.train_news['Statement'][:10000], prep.train_news['Label'][:10000])

gs_clf.best_score_
gs_clf.best_params_
gs_clf.cv_results_

# logistic regression parameters
parameters = {'LogR_tfidf__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)],
              'LogR_tfidf__use_idf': (True, False),
              'LogR_tfidf__smooth_idf': (True, False)
              }

gs_clf = GridSearchCV(logR_pipeline_ngram, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(prep.train_news['Statement'][:10000], prep.train_news['Label'][:10000])

gs_clf.best_score_
gs_clf.best_params_
gs_clf.cv_results_

# Linear SVM parameters
parameters = {'svm_tfidf__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)],
              'svm_tfidf__use_idf': (True, False),
              'svm_tfidf__smooth_idf': (True, False),
              'svm_clf__penalty': ('l1', 'l2'),
              }

gs_clf = GridSearchCV(svm_pipeline_ngram, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(prep.train_news['Statement'][:10000], prep.train_news['Label'][:10000])

gs_clf.best_score_
gs_clf.best_params_
gs_clf.cv_results_

# GridSearch
random_forest_final = Pipeline([
    ('rf_tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 3), use_idf=True, smooth_idf=True)),
    ('rf_clf', RandomForestClassifier(n_estimators=300, n_jobs=3, max_depth=10))
])

random_forest_final.fit(prep.train_news['Statement'], prep.train_news['Label'])
predicted_rf_final = random_forest_final.predict(prep.test_news['Statement'])
np.mean(predicted_rf_final == prep.test_news['Label'])
print(metrics.classification_report(prep.test_news['Label'], predicted_rf_final))

logR_pipeline_final = Pipeline([
    ('LogR_tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 5), use_idf=True, smooth_idf=False)),
    ('LogR_clf', LogisticRegression(penalty="l2", C=1))
])

logR_pipeline_final.fit(prep.train_news['Statement'], prep.train_news['Label'])
predicted_LogR_final = logR_pipeline_final.predict(prep.test_news['Statement'])
np.mean(predicted_LogR_final == prep.test_news['Label'])
print(metrics.classification_report(prep.test_news['Label'], predicted_LogR_final))

# saving best model
model_file = 'final_model.sav'
pickle.dump(logR_pipeline_ngram, open(model_file, 'wb'))

# plot learning curve
def plot_learing_curve(pipeline, title):
    size = 10000
    cv = KFold(size, shuffle=True)

    X = prep.train_news["Statement"]
    y = prep.train_news["Label"]

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

# plotting Precision-Recall curve
def plot_PR_curve(classifier):
    precision, recall, thresholds = precision_recall_curve(prep.test_news['Label'], classifier)
    average_precision = average_precision_score(prep.test_news['Label'], classifier)

    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Random Forest Precision-Recall curve: AP={0:0.2f}'.format(
        average_precision))


plot_PR_curve(predicted_LogR_ngram)
plot_PR_curve(predicted_rf_ngram)

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