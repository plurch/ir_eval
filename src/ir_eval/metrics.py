
# Resources:
# https://www.pinecone.io/learn/offline-evaluation/

def recall(actual: list[int], predicted: list[int], k: int) -> float:
  """
  Calculate the recall@k metric.

  Recall is defined as the ratio of the total number of relevant documents retrieved
  within the top-k predictions to the total number of relevant documents in the database.

  Recall =  Total number of documents retrieved that are relevant/Total number of relevant documents in the database.

  Parameters:
    actual (list[int]): An array of ground truth relevant items.
    predicted (list[int]): An array of predicted items, ordered by relevance.
    k (int): The number of top predictions to consider.

  Returns:
    float: The recall value at rank k, ranging from 0 to 1.
           A value of 1 indicates perfect recall, while 0 indicates no relevant documents retrieved.

  Example:
    >>> actual = [1, 2, 3, 4]
    >>> predicted = [4, 2, 6, 1, 7]
    >>> k = 3
    >>> recall(actual, predicted, k)
    0.5  # (2 relevant documents retrieved out of 4 total in dataset)

  Notes:
    - This function assumes the `predicted` array is sorted in descending order of relevance.
    - If k is larger than the length of the `predicted` array, it will consider the entire array.
  """
  actual_set = set(actual)
  top_k_predictions = set(predicted[:k])
  count_relevant_in_top_k = len(actual_set.intersection(top_k_predictions))
  return count_relevant_in_top_k / float(len(actual_set))

def precision(actual: list[int], predicted: list[int], k: int) -> float:
  """
  Calculate the precision@k metric.

  Precision is defined as the ratio of the total number of relevant documents retrieved
  within the top-k predictions to the total number of returned documents (k).

  Precision =  Total number of documents retrieved that are relevant/Total number of documents that are retrieved.

  Parameters:
    actual (list[int]): An array of ground truth relevant items.
    predicted (list[int]): An array of predicted items, ordered by relevance.
    k (int): The number of top predictions to consider.

  Returns:
    float: The precision value at rank k, ranging from 0 to 1.
           A value of 1 indicates perfect precision, while 0 indicates no relevant documents retrieved.

  Example:
    >>> actual = [1, 2, 3, 4]
    >>> predicted = [4, 2, 6, 1, 7]
    >>> k = 3
    >>> precision(actual, predicted, k)
    0.66  # (2 relevant documents retrieved out of 3 returned)

  Notes:
    - This function assumes the `predicted` array is sorted in descending order of relevance.
    - If k is larger than the length of the `predicted` array, it will consider the entire array.
  """
  actual_set = set(actual)
  top_k_predictions = set(predicted[:k])
  count_relevant_in_top_k = len(actual_set.intersection(top_k_predictions))
  return count_relevant_in_top_k / float(k)


def average_precision(actual: list[int], predicted: list[int], k: int) -> float:
  """
  Computes the Average Precision (AP) at a specified rank `k`.

  Average Precision (AP) is a metric used to evaluate the relevance of predicted rankings 
  in information retrieval tasks. It is calculated as the mean of precision values at 
  each rank where a relevant document is retrieved within the top `k` predictions.

  Args:
      actual (list[int]): A list of integers representing the ground truth relevant items.
      predicted (list[int]): A list of integers representing the predicted rankings of items.
      k (int): The maximum number of top-ranked items to consider for evaluation.

  Returns:
      float: The Average Precision score. If no relevant items are retrieved within the
      top `k` predictions, the function may raise a division by zero error or return `NaN`.

  Example:
      >>> actual = [1, 2, 3]
      >>> predicted = [1, 4, 2, 3]
      >>> k = 3
      >>> average_precision(actual, predicted, k)
      0.7777777777777778  # Example AP score.
  """
  actual_set = set(actual)
  precision_list = []

  for i in range(k):
    if (predicted[i] in actual_set):
      precision_list.append(precision(actual, predicted, i+1))

  return sum(precision_list) / len(precision_list)
