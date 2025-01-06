import pytest
from ir_eval.metrics import recall, precision, average_precision, mean_average_precision

# Sample data generated with:
# total_count_items = 100
# total_relevant_items = 25
# rng = np.random.default_rng()
# actual = rng.choice(total_count_items, total_relevant_items, replace=False)

actual = [ 4, 79, 32, 45, 14, 46, 53, 15,  3, 54, 68, 99, 75, 82, 35, 27, 73,
    20, 25, 66, 11, 58, 31,  8, 85]
predicted = [1, 2, 62, 84, 3, 4, 81, 14, 5, 67]
# intersection: {3, 4, 14}

class TestRecall:
  def test_recall_k_5(self):
    result = recall(actual, predicted, 5)
    assert result == pytest.approx(0.04) # 1 out of 25
  
  def test_recall_k_10(self):
    result = recall(actual, predicted, 10)
    assert result == pytest.approx(0.12) # 3 out of 25

class TestPrecision:
  def test_precision_k_5(self):
    result = precision(actual, predicted, 5)
    assert result == pytest.approx(0.2) # 1 out of 5
  
  def test_precision_k_10(self):
    result = precision(actual, predicted, 10)
    assert result == pytest.approx(0.3) # 3 out of 10

class TestAveragePrecision:
  def test_average_precision_basic(self):
    # basic inputs
    result = average_precision([1,3,5], [1,2,3,4,5], 5)
    assert result == pytest.approx(0.7555555555555555) # (1 + 0.67 + 0.6) / 3 = 0.75555

  def test_precision_k_5(self):
    result = average_precision(actual, predicted, 5)
    assert result == pytest.approx(0.2)

  def test_precision_k_10(self):
    result = average_precision(actual, predicted, 10)
    assert result == pytest.approx(0.30277777777777776)

class TestMeanAveragePrecision:
  def test_mean_average_precision_basic(self):
    # basic inputs
    actual_list = [
      [1,3,5],
      [2,4,6],
      [7,8,9]
    ]

    predicted_list = [
      [1,2,3,4,5],
      [9,2,3,1,5],
      [4,5,9,8,3]
    ]
    result = mean_average_precision(actual_list, predicted_list, 5)
    # ap values: [0.7555555555555555, 0.5, 0.41666666666666663]
    assert result == pytest.approx(0.5574074074074074)

  def test_mean_average_precision_pydoc(self):
    # inputs from pydoc string
    actual_list = [[1, 2, 3], [2, 3, 4]]

    predicted_list = [[1, 4, 2, 3], [2, 3, 5, 4]]
    result = mean_average_precision(actual_list, predicted_list, 3)
    # ap values: [0.8333333333333333, 1.0]
    assert result == pytest.approx(0.9166666666666666)
