# Semantic Analysis of Movie Reviews using Character N-gram

In this project, semantic analysis of movie reviews is done by using character [n-gram](https://github.com/halfvim/N-Gram/wiki/N-gram). The review data is in the folder `data`. Reviews in `data/pos` are semantically positive, while reviews in `data/neg` are semantically negative. [Leave-one-out cross validation](https://github.com/halfvim/N-Gram/wiki/Leave-one-out-Cross-Validation) is implemented to test the accuracy of both [out-of-place distance measure](https://github.com/halfvim/N-Gram/wiki/Out-of-place-Distance-Measure) and [Naive Bayes classifier](https://github.com/halfvim/N-Gram/wiki/Naive-Bayes-Classifier).

An introduction of this project can be found at the [wiki](https://github.com/halfvim/N-Gram/wiki/Introduction) page.

This project is implemented in Python 2.

To run the program, open terminal (console) in Mac OS X or Linux (Windows), change the directory to the project directory, then use the following command to run the program:
```
python ngram.py alg gram-n start end
```

There are four parameters in this command: `alg`, `gram-n`, `start`, and `end`:

-	`alg`: the algorithm chosen, could be "oop" or "nb". "oop" represents out-of-place measure, while "nb" represents Naive Bayes classifier.
-	`gram-n`: the value of N in N-gram, could be any positive integer. However, integers between 2 and 9 is recommanded.
-	`start`, `end`: the start and end of the leave-one-out cross validation. Start has to be smaller than end. Start and end should be in the range of [0, 2000].

For example, to run 6-gram out-of-place algorithm with full cross validation, you should use:
```
python ngram.py oop 6 0 2000
```

To run 3-gram Naive Bayes algorithm with part of the cross validation, you should use:
```
python ngram.py nb 3 100 200
```

The accuracy results of the two algorithm is [here](https://github.com/halfvim/N-Gram/wiki/Results).
