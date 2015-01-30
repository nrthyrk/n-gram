# Semantic Analysis of Movie Reviews using Character N-gram

In this project, semantic analysis of movie reviews is done by using character n-gram. The review data is in the folder `data`. Reviews in `data/pos` are semantically positive, while reviews in `data/neg` are semantically negative. Leave-one-out cross validation is implemented to test the accuracy of both `out-of-place measure` and `Naive Bayes classifier`.

This project is implemented in Python. To run this program, you need Python 2.7 installed.

To run the program, open terminal in Mac OS X or Linux, open console window in Windows, change the directory to the project directory.

Then use the following command to run the program:
```
python ngram.py alg gram-n start end
```

There are four parameters in this command: `alg`, `gram-n`, `start`, and `end`. They have the following meanings:

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
