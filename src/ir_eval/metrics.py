import numpy as np
import numpy.typing as npt

# Resources:
# https://www.pinecone.io/learn/offline-evaluation/

def recall(actual: npt.NDArray, predicted: npt.NDArray, k: int) -> float:
  """
  Calculate the recall@k metric.

  Recall is defined as the ratio of the total number of relevant documents retrieved
  within the top-k predictions to the total number of relevant documents in the database.

  Recall =  Total number of documents retrieved that are relevant/Total number of relevant documents in the database.

  Parameters:
    actual (npt.NDArray): An array of ground truth relevant items.
    predicted (npt.NDArray): An array of predicted items, ordered by relevance.
    k (int): The number of top predictions to consider.

  Returns:
    float: The recall value at rank k, ranging from 0 to 1.
           A value of 1 indicates perfect recall, while 0 indicates no relevant documents retrieved.

  Example:
    >>> actual = np.array([1, 2, 3, 4])
    >>> predicted = np.array([4, 2, 6, 1, 7])
    >>> k = 3
    >>> recall(actual, predicted, k)
    0.5  # (2 relevant documents retrieved out of 4)

  Notes:
    - This function assumes the `predicted` array is sorted in descending order of relevance.
    - If k is larger than the length of the `predicted` array, it will consider the entire array.
  """
  actual_set = set(actual)
  top_k_predictions = set(predicted[:k])
  count_relevant_in_top_k = len(actual_set.intersection(top_k_predictions))
  return count_relevant_in_top_k / float(len(actual_set))

def precision(actual: npt.NDArray, predicted: npt.NDArray, k: int) -> float:
  """
  Calculate the precision@k metric.

  Precision is defined as the ratio of the total number of relevant documents retrieved
  within the top-k predictions to the total number of returned documents (k).

  Precision =  Total number of documents retrieved that are relevant/Total number of documents that are retrieved.

  Parameters:
    actual (npt.NDArray): An array of ground truth relevant items.
    predicted (npt.NDArray): An array of predicted items, ordered by relevance.
    k (int): The number of top predictions to consider.

  Returns:
    float: The precision value at rank k, ranging from 0 to 1.
           A value of 1 indicates perfect precision, while 0 indicates no relevant documents retrieved.

  Example:
    >>> actual = np.array([1, 2, 3, 4])
    >>> predicted = np.array([4, 2, 6, 1, 7])
    >>> k = 3
    >>> precision(actual, predicted, k)
    0.5  # (2 relevant documents retrieved out of 4)

  Notes:
    - This function assumes the `predicted` array is sorted in descending order of relevance.
    - If k is larger than the length of the `predicted` array, it will consider the entire array.
  """
  actual_set = set(actual)
  top_k_predictions = set(predicted[:k])
  count_relevant_in_top_k = len(actual_set.intersection(top_k_predictions))
  return count_relevant_in_top_k / float(k)
