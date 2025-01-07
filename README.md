# Information Retrieval Evaluation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`ir_eval` - Information retrieval evaluation metrics in pure python with zero dependencies

This project provides simple and tested python implementations of information retrieval metrics without any library dependencies (not even numpy!). The source code is clear and easy to understand. All functions have pydoc help strings.

The metrics can be used to determine the quality of rankings that are returned by a retrieval or recommender system.

## Installation

`ir_eval` can be installed from pypi with:

```
pip install ir_eval
```

## Usage

Metric functions will generally accept the following arguments:

`actual` (list[int]): An array of ground truth relevant items.

`predicted` (list[int]): An array of predicted items, ordered by relevance.

`k` (int): The number of top predictions to consider.

Functions will return a `float` value as the computed metric value.

## Unit tests

Unit tests with easy to follow scenarios and sample data are included.

### Run unit tests
```
uv run pytest
```

## Metrics

### Recall

Recall is defined as the ratio of the total number of relevant items retrieved within the top-k predictions to the total number of relevant items in the entire database.

```
from ir_eval.metrics import recall
```

### Precision

Precision is defined as the ratio of the total number of relevant items retrieved  within the top-k predictions to the total number of returned items (k).

```
from ir_eval.metrics import precision
```

### Average Precision (AP)

Average Precision is calculated as the mean of precision values at  each rank where a relevant item is retrieved within the top `k` predictions.

```
from ir_eval.metrics import average_precision
```

### Mean Average Precision (MAP)

MAP is the mean of the Average Precision (AP - see above) scores computed for multiple queries.

```
from ir_eval.metrics import mean_average_precision
```

### Normalized Discounted Cumulative Gain (nDCG)

nDCG evaluates the quality of a predicted ranking by comparing it to an ideal ranking (i.e., perfect ordering of relevant items). It accounts for the position of relevant items in the ranking, giving higher weight to items appearing earlier.

```
from ir_eval.metrics import ndcg
```

### Reciprocal Rank (RR)

Reciprocal Rank (RR) assigns a score based on the reciprocal of the rank at which the first relevant item is found.

```
from ir_eval.metrics import reciprocal_rank
```

### Mean Reciprocal Rank (MRR)

MRR calculates the mean of the Reciprocal Rank (RR) scores for a set of queries.

```
from ir_eval.metrics import mean_reciprocal_rank
```

## Online Resources

[Pinecone - Evaluation Measures in Information Retrieval
](https://www.pinecone.io/learn/offline-evaluation/)

[Spot Intelligence - Mean Average Precision](https://spotintelligence.com/2023/09/07/mean-average-precision/)

[Spot Intelligence - Mean Reciprocal Rank](https://spotintelligence.com/2024/08/02/mean-reciprocal-rank-mrr/)