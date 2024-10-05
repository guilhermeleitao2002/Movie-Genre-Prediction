# Progress Report
## From & Plot + Lemmatization + Stopwords("english")
### Results
```
Naive Bayes Accuracy: 53.70%
SVM Accuracy: 65.32%
```

## From & Plot + Lemmatization + Stopwords("and,his,in,is,of,the,to,with")
### Results
```
Naive Bayes Accuracy: 59.54%
SVM Accuracy: 65.61%
```

## List of experiments and respective results with lemmatization and stopwords 'english'
```	
Running experiment with max_features=4000, ngram_range=(1,2) and combine_fields=from,director,title
Finished. Took 42.2 seconds. Naive Bayes Accuracy: 60.22%, SVM Accuracy: 65.88%
Running experiment with max_features=4000, ngram_range=(1,3) and combine_fields=from,director,title
Finished. Took 45.52 seconds. Naive Bayes Accuracy: 60.10%, SVM Accuracy: 66.00%
Running experiment with max_features=5000, ngram_range=(1,2) and combine_fields=from,director,title
Finished. Took 48.21 seconds. Naive Bayes Accuracy: 59.60%, SVM Accuracy: 66.13%
Running experiment with max_features=5000, ngram_range=(1,3) and combine_fields=from,director,title
Finished. Took 48.39 seconds. Naive Bayes Accuracy: 59.54%, SVM Accuracy: 66.07%
Running experiment with max_features=6000, ngram_range=(1,2) and combine_fields=from,director,title
Finished. Took 47.65 seconds. Naive Bayes Accuracy: 58.79%, SVM Accuracy: 65.69%
Running experiment with max_features=6000, ngram_range=(1,3) and combine_fields=from,director,title
Finished. Took 53.66 seconds. Naive Bayes Accuracy: 58.79%, SVM Accuracy: 65.63%
Running experiment with max_features=4000, ngram_range=(1,2) and combine_fields=title
Finished. Took 41.71 seconds. Naive Bayes Accuracy: 58.42%, SVM Accuracy: 64.08%
Running experiment with max_features=4000, ngram_range=(1,3) and combine_fields=title
Finished. Took 44.72 seconds. Naive Bayes Accuracy: 58.61%, SVM Accuracy: 63.77%
Running experiment with max_features=5000, ngram_range=(1,2) and combine_fields=title
Finished. Took 47.27 seconds. Naive Bayes Accuracy: 58.36%, SVM Accuracy: 62.71%
Running experiment with max_features=5000, ngram_range=(1,3) and combine_fields=title
Finished. Took 48.21 seconds. Naive Bayes Accuracy: 58.36%, SVM Accuracy: 62.71%
Running experiment with max_features=6000, ngram_range=(1,2) and combine_fields=title
Finished. Took 46.62 seconds. Naive Bayes Accuracy: 57.80%, SVM Accuracy: 63.58%
Running experiment with max_features=6000, ngram_range=(1,3) and combine_fields=title
Finished. Took 52.73 seconds. Naive Bayes Accuracy: 57.74%, SVM Accuracy: 63.70%
Running experiment with max_features=4000, ngram_range=(1,2) and combine_fields=from
Finished. Took 42.05 seconds. Naive Bayes Accuracy: 60.16%, SVM Accuracy: 65.88%
Running experiment with max_features=4000, ngram_range=(1,3) and combine_fields=from
Finished. Took 45.19 seconds. Naive Bayes Accuracy: 60.16%, SVM Accuracy: 65.94%
Running experiment with max_features=5000, ngram_range=(1,2) and combine_fields=from
Finished. Took 47.61 seconds. Naive Bayes Accuracy: 59.48%, SVM Accuracy: 65.38%
Running experiment with max_features=5000, ngram_range=(1,3) and combine_fields=from
Finished. Took 47.51 seconds. Naive Bayes Accuracy: 59.48%, SVM Accuracy: 65.20%
Running experiment with max_features=6000, ngram_range=(1,2) and combine_fields=from
Finished. Took 47.17 seconds. Naive Bayes Accuracy: 58.79%, SVM Accuracy: 64.64%
Running experiment with max_features=6000, ngram_range=(1,3) and combine_fields=from
Finished. Took 53.97 seconds. Naive Bayes Accuracy: 58.86%, SVM Accuracy: 64.76%
Running experiment with max_features=4000, ngram_range=(1,2) and combine_fields=director
Finished. Took 42.42 seconds. Naive Bayes Accuracy: 59.23%, SVM Accuracy: 63.64%
Running experiment with max_features=4000, ngram_range=(1,3) and combine_fields=director
Finished. Took 48.02 seconds. Naive Bayes Accuracy: 59.04%, SVM Accuracy: 63.95%
Running experiment with max_features=5000, ngram_range=(1,2) and combine_fields=director
Finished. Took 45.77 seconds. Naive Bayes Accuracy: 58.61%, SVM Accuracy: 63.64%
Running experiment with max_features=5000, ngram_range=(1,3) and combine_fields=director
Finished. Took 49.83 seconds. Naive Bayes Accuracy: 58.48%, SVM Accuracy: 63.64%
Running experiment with max_features=6000, ngram_range=(1,2) and combine_fields=director
Finished. Took 50.9 seconds. Naive Bayes Accuracy: 58.11%, SVM Accuracy: 63.95%
Running experiment with max_features=6000, ngram_range=(1,3) and combine_fields=director
Finished. Took 51.52 seconds. Naive Bayes Accuracy: 58.11%, SVM Accuracy: 63.83%
Running experiment with max_features=4000, ngram_range=(1,2) and combine_fields=from,director
Finished. Took 42.75 seconds. Naive Bayes Accuracy: 60.22%, SVM Accuracy: 65.63%
Running experiment with max_features=4000, ngram_range=(1,3) and combine_fields=from,director
Finished. Took 49.17 seconds. Naive Bayes Accuracy: 60.29%, SVM Accuracy: 65.69%
Running experiment with max_features=5000, ngram_range=(1,2) and combine_fields=from,director
Finished. Took 46.07 seconds. Naive Bayes Accuracy: 59.54%, SVM Accuracy: 65.51%
Running experiment with max_features=5000, ngram_range=(1,3) and combine_fields=from,director
Finished. Took 51.76 seconds. Naive Bayes Accuracy: 59.48%, SVM Accuracy: 65.26%
```