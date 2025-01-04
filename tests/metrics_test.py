import pytest
import numpy as np

from ir_eval.metrics import recall, precision

# Sample data generated with:
# total_count_items = 100
# total_relevant_items = 25
# rng = np.random.default_rng()
# actual = rng.choice(total_count_items, total_relevant_items, replace=False)

actual = np.array([ 4, 79, 32, 45, 14, 46, 53, 15,  3, 54, 68, 99, 75, 82, 35, 27, 73,
    20, 25, 66, 11, 58, 31,  8, 85])
predicted = np.array([1, 2, 62, 84, 3, 4, 81, 14, 5, 67])

class TestRecall:
  def test_recall_k_5(self):
    result = recall(actual, predicted, 5)
    assert result == pytest.approx(0.04)
  
  def test_recall_k_10(self):
    result = recall(actual, predicted, 10)
    assert result == pytest.approx(0.12)

class TestPrecision:
  def test_precision_k_5(self):
    result = precision(actual, predicted, 5)
    assert result == pytest.approx(0.2)
  
  def test_precision_k_10(self):
    result = precision(actual, predicted, 10)
    assert result == pytest.approx(0.3)
