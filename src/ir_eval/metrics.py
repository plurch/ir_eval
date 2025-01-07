import math

# Resources:
# https://www.pinecone.io/learn/offline-evaluation/
# https://spotintelligence.com/2023/09/07/mean-average-precision/
# https://spotintelligence.com/2024/08/02/mean-reciprocal-rank-mrr/

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
  """
  actual_set = set(actual)
  precision_list = []

  for i in range(k):
    if (predicted[i] in actual_set):
      precision_list.append(precision(actual, predicted, i+1))

  return sum(precision_list) / len(precision_list)

def mean_average_precision(actual_list: list[list[int]], predicted_list: list[list[int]], k: int) -> float:
  """
  Computes the Mean Average Precision (MAP) at a specified rank `k`.

  It is the mean of the Average Precision (AP) scores computed for multiple 
  queries

  Args:
      actual_list (list[list[int]]): A list of lists where each inner list represents 
          the ground truth relevant items for a query
      predicted_list (list[list[int]]): A list of lists where each inner list represents 
          the predicted rankings of items for a query
      k (int): The maximum number of top-ranked items to consider for each prediction.

  Returns:
      float: The Mean Average Precision score, which is the mean of AP scores across all 
      queries.

  Raises:
      AssertionError: If the lengths of `actual_list` and `predicted_list` are not equal.
  """

  assert len(actual_list) == len(predicted_list)

  ap_values = [average_precision(actual_list[i], predicted_list[i], k) for i in range(len(actual_list))]
  return sum(ap_values) / len(ap_values)

def ndcg(actual: list[int], predicted: list[int], k: int) -> float:
  """
  Computes the Normalized Discounted Cumulative Gain (nDCG) at a specified rank `k`.

  It evaluates the quality of a predicted ranking by comparing it to an ideal ranking 
  (i.e., perfect ordering of relevant items). It accounts for the position of relevant 
  items in the ranking, giving higher weight to items appearing earlier.

  Args:
      actual (list[int]): A list of integers representing the ground truth relevant items.
      predicted (list[int]): A list of integers representing the predicted rankings of items.
      k (int): The maximum number of top-ranked items to consider for evaluation.

  Returns:
      float: The nDCG score, which ranges from 0 to 1. A value of 1 indicates a perfect 
      ranking. Returns 0 if there are no relevant items in the top `k` predictions or if 
      the ideal DCG (iDCG) is zero.
  """
  actual_set = set(actual)

  # discounted cumulative gain
  # `i+2` due to zero indexing
  dcg = sum([1.0/math.log2(i+2) if predicted[i] in actual_set else 0 for i in range(k)])
  # ideal discounted cumulative gain (ie. perfect results returned)
  idcg = sum([1.0/math.log2(i+2) for i in range(min(k, len(actual_set)))])
  return dcg / idcg

def reciprocal_rank(actual: list[int], predicted: list[int], k: int) -> float:
  """
  Computes the Reciprocal Rank (RR) at a specified rank `k`.

  It assigns a score based on the reciprocal of the rank at which the first relevant item is found.

  Args:
      actual (list[int]): A list of integers representing the ground truth relevant items.
      predicted (list[int]): A list of integers representing the predicted rankings of items.
      k (int): The maximum number of top-ranked items to consider for evaluation.

  Returns:
      float: The Reciprocal Rank score. A value of 0 is returned if no relevant items are 
      found within the top `k` predictions. Otherwise, the score is `1 / rank`, where 
      `rank` is the position (1-based) of the first relevant item.

  Notes:
      - The function assumes zero-based indexing for the `predicted` list.
      - If `k` exceeds the length of `predicted`, only the available elements in `predicted` 
        are considered.
      - This metric focuses only on the rank of the first relevant item, ignoring others.
  """
  actual_set = set(actual)

  for i in range(k):
    if predicted[i] in actual_set:
      return 1 / float(i + 1)
  
  return float(0)

def mean_reciprocal_rank(actual_list: list[list[int]], predicted_list: list[list[int]], k: int) -> float:
  """
  Computes the Mean Reciprocal Rank (MRR) at a specified rank `k`.

  It calculates the mean of the Reciprocal Rank (RR) scores for a set of queries.

  Args:
      actual_list (list[list[int]]): A list of lists where each inner list represents the 
          ground truth relevant items for a query or task.
      predicted_list (list[list[int]]): A list of lists where each inner list represents 
          the predicted rankings of items for a query or task.
      k (int): The maximum number of top-ranked items to consider for each prediction.

  Returns:
      float: The Mean Reciprocal Rank score, which is the average of RR scores across all 
      queries. Returns 0 if there are no queries or no relevant items in any predictions.

  Raises:
      AssertionError: If the lengths of `actual_list` and `predicted_list` are not equal.
  """

  assert len(actual_list) == len(predicted_list)

  rr_values = [reciprocal_rank(actual_list[i], predicted_list[i], k) for i in range(len(actual_list))]
  return sum(rr_values) / len(rr_values)
