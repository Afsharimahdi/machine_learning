# Machine Learning Repository

Welcome to the Machine Learning Repository! This repository is a collection of various machine learning projects, algorithms, and resources. Whether you're a beginner or an experienced practitioner, this repository provides a wide range of materials to help you learn and apply machine learning concepts effectively.

## Table of Contents

1. [Projects](#projects)
2. [Algorithms](#algorithms)
3. [Resources](#resources)
4. [Contributing](#contributing)
5. [License](#license)

## Projects

This section includes different machine learning projects that cover various domains and applications. Each project comes with its own documentation, code, and datasets. Feel free to explore and experiment with these projects:

<!-- Add project links here -->

## Algorithms

This section provides implementation examples and resources for popular machine learning algorithms. Each algorithm is explained in detail and comes with code samples and practical use cases:

1. [k-nearest neighbors algorithm](./algorithms/linear-regression/README.md)
2. [Naive Bayes](https://github.com/Afsharimahdi/machine_learning/blob/master/Gaussian%20Naive%20Bayes.md)
3. []()
4. []()
5. []()
6. []()

<!-- Add more algorithms here -->

## Resources

This section contains additional resources and references for learning machine learning concepts, techniques, and tools. It includes books, tutorials, online courses, and research papers to deepen your understanding and improve your skills.

<!-- Add resource links here -->

## Contributing

Contributions to this repository are welcome! If you have a machine learning project, algorithm implementation, or useful resource that you'd like to share with the community, please follow the guidelines in the [CONTRIBUTING.md](./CONTRIBUTING.md) file.

## License

This repository is licensed under the [MIT License](./LICENSE). Feel free to use the code and resources provided in this repository for your own learning and projects.


## Resources

This section contains additional resources, tutorials, and articles related to machine learning. It provides a wealth of information to help you deepen your understanding and enhance your machine learning skills:

1. [Books](./resources/books.md): A curated list of recommended books for machine learning enthusiasts.
2. [Online Courses](./resources/online-courses.md): A collection of online courses and tutorials to learn machine learning.
3. [Blogs](./resources/blogs.md): A list of popular blogs and websites for staying up to date with machine learning advancements.
4. [Datasets](./resources/datasets.md): A compilation of publicly available datasets for practicing machine learning.

## Contributing

We welcome contributions to this repository! If you have any machine learning projects, algorithms, or resources that you would like to add, please follow the contribution guidelines outlined in the [CONTRIBUTING](./CONTRIBUTING.md) file.

## License

This repository is licensed under the MIT License. For more details, please refer to the [LICENSE](./LICENSE) file.

Happy machine learning!


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
