## Gaussian_Naive_Bayes

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Implementation of the Gaussian Naive Bayes algorithm from scratch using NumPy.

This project provides a Python implementation of the Gaussian Naive Bayes algorithm, which is a simple yet effective classifier for classification tasks. The implementation is built from scratch using NumPy.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Example](#example)
- [Contributing](#contributing)
- [License](#license)

## Installation

To use this implementation, you need to have NumPy and scikit-learn installed. You can install them using pip:

```shell
pip install numpy scikit-learn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
import time

class GaussianNaiveBayes:
    # GaussianNaiveBayes implementation code

# Load example dataset
X, y = datasets.make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

start = time.perf_counter()
nb = GaussianNaiveBayes()
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)
end = time.perf_counter()
print(f"NumPy Naive Bayes accuracy: {accuracy_score(y_test, predictions)}")
print(f'Finished in {round(end - start, 3)} second(s)')

from sklearn.naive_bayes import GaussianNB

start = time.perf_counter()
sk_nb = GaussianNB()
sk_nb.fit(X_train, y_train)
sk_predictions = sk_nb.predict(X_test)
end = time.perf_counter()
print(f"scikit-learn Naive Bayes accuracy: {accuracy_score(y_test, sk_predictions)}")
print(f'Finished in {round(end - start, 3)} second(s)')
