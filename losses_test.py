# Copyright 2019 The TensorFlow Ranking Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for ranking losses."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import math
import losses as ranking_losses
import losses_impl
import unittest


def ln(x):
    return math.log(x)


def allClose(input, actual):
    eps = 1.e-4
    assert input.shape == actual.shape
    ft1, ft2 = input.flatten(), actual.flatten()
    for i in range(len(ft1)):
        if not ((math.fabs(ft2[i]) - eps) < math.fabs(ft1[i]) < (math.fabs(ft2[i]) + eps)):
            print("input is ", input, " actual is ", actual)
            return False
    return True


class DCGLambdaWeightTest(unittest.TestCase):
    """Test cases for DCGLambdaWeight."""

    def test_default(self):
        """For the weight using rank diff."""
        labels = torch.tensor([[2.0, 1.0, 0.0]])
        ranks = torch.tensor([[1, 2, 3]])
        lambda_weight = losses_impl.DCGLambdaWeight()
        self.assertTrue(allClose(
            lambda_weight.pair_weights(labels, ranks),
            torch.tensor([[[0., 1. / 2., 2. * 1. / 6.], [1. / 2., 0., 1. / 2.],
                           [2. * 1. / 6., 1. / 2., 0.]]])))

    def test_smooth_fraction(self):
        """For the weights using absolute rank."""
        labels = torch.tensor([[2.0, 1.0, 0.0]])
        ranks = torch.tensor([[1, 2, 3]])
        actual1 = torch.tensor([[[0., 1. / 2., 2. * 2. / 3.], [1. / 2., 0., 1. / 6.], [2. * 2. / 3., 1. / 6., 0.]]])
        actual2 = torch.tensor([[[0., 1., 2.], [1., 0., 0.], [2., 0., 0.]]])
        lambda_weight = losses_impl.DCGLambdaWeight(smooth_fraction=1.0)
        self.assertTrue(allClose(lambda_weight.pair_weights(labels, ranks), actual1))
        lambda_weight = losses_impl.DCGLambdaWeight(topn=1, smooth_fraction=1.0)
        self.assertTrue(allClose(lambda_weight.pair_weights(labels, ranks), actual2))

    def test_topn(self):
        labels = torch.tensor([[2.0, 1.0, 0.0]])
        ranks = torch.tensor([[1, 2, 3]])
        actual = torch.tensor([[[0., 1. / 2., 2. * 1. / 2.], [1. / 2., 0., 0.],
                                [2. * 1. / 2., 0., 0.]]])
        lambda_weight = losses_impl.DCGLambdaWeight(topn=1)
        self.assertTrue(allClose(lambda_weight.pair_weights(labels, ranks), actual))

    def test_invalid_labels(self):
        labels = torch.tensor([[2.0, 1.0, -1.0]])
        ranks = torch.tensor([[1, 2, 3]])
        actual = torch.tensor([[[0., 1. / 2., 0.], [1. / 2., 0., 0.], [0., 0., 0.]]])
        lambda_weight = losses_impl.DCGLambdaWeight()
        self.assertTrue(allClose(lambda_weight.pair_weights(labels, ranks), actual))

    def test_gain_and_discount(self):
        labels = torch.tensor([[2.0, 1.0]])
        ranks = torch.tensor([[1, 2]])
        actual = torch.tensor([[[0., 2. * (1. / math.log(2.) - 1. / math.log(3.))],
                                [2. * (1. / math.log(2.) - 1. / math.log(3.)), 0.]]])
        lambda_weight = losses_impl.DCGLambdaWeight(
            gain_fn=lambda x: torch.pow(2., x) - 1.,
            rank_discount_fn=lambda r: 1. / torch.log1p(r))
        self.assertTrue(allClose(
            lambda_weight.pair_weights(labels, ranks), actual))

    def test_normalized(self):
        labels = torch.tensor([[2.0, 1.0]])
        ranks = torch.tensor([[1, 2]])
        lambda_weight = losses_impl.DCGLambdaWeight(normalized=True)
        max_dcg = 2.5
        self.assertTrue(allClose(
            lambda_weight.pair_weights(labels, ranks),
            torch.tensor([[[0., 1. / 2. / max_dcg], [1. / 2. / max_dcg, 0.]]])))

    def test_individual_weights(self):
        labels = torch.tensor([[1.0, 2.0]])
        ranks = torch.tensor([[1, 2]])
        lambda_weight = losses_impl.DCGLambdaWeight(normalized=True)
        max_dcg = 2.5
        self.assertTrue(allClose(lambda_weight.individual_weights(labels, ranks),
                                 torch.tensor([[1. / max_dcg / 1., 2. / max_dcg / 2.]])))

    def test_create_ndcg_lambda_weight(self):
        labels = torch.tensor([[2.0, 1.0]])
        ranks = torch.tensor([[1, 2]])
        lambda_weight = ranking_losses.create_ndcg_lambda_weight()
        max_dcg = 3.0 / math.log(2.) + 1.0 / math.log(3.)
        self.assertTrue(allClose(
            lambda_weight.pair_weights(labels, ranks),
            torch.tensor([[[0., 2. * (1. / math.log(2.) - 1. / math.log(3.)) / max_dcg],
                           [2. * (1. / math.log(2.) - 1. / math.log(3.)) / max_dcg, 0.]]])))

    def test_create_reciprocal_rank_lambda_weight(self):
        labels = torch.tensor([[2.0, 1.0]])
        ranks = torch.tensor([[1, 2]])
        lambda_weight = ranking_losses.create_reciprocal_rank_lambda_weight()
        max_dcg = 2.5
        self.assertTrue(allClose(
            lambda_weight.pair_weights(labels, ranks),
            torch.tensor([[[0., 1. / 2. / max_dcg], [1. / 2. / max_dcg, 0.]]])))

    def test_create_p_list_mle_lambda_weight(self):
        labels = torch.tensor([[2.0, 1.0]])
        ranks = torch.tensor([[1, 2]])
        lambda_weight = ranking_losses.create_p_list_mle_lambda_weight(2)
        self.assertTrue(allClose(
            lambda_weight.individual_weights(labels, ranks),
            torch.tensor([[1.0, 0.0]])))


class PrecisionLambdaWeightTest(unittest.TestCase):
    """Test cases for PrecisionLambdaWeight."""

    def test_default(self):
        labels = torch.tensor([[2.0, 1.0, 0.0]])
        ranks = torch.tensor([[1, 2, 3]])
        lambda_weight = losses_impl.PrecisionLambdaWeight(topn=5)
        self.assertTrue(allClose(
                lambda_weight.pair_weights(labels, ranks),
                torch.tensor([[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]])))

        lambda_weight = losses_impl.PrecisionLambdaWeight(topn=1)
        self.assertTrue(allClose(
                lambda_weight.pair_weights(labels, ranks),
                torch.tensor([[[0., 0., 1.], [0., 0., 0.], [1., 0., 0.]]])))




def _pairwise_loss(labels, scores, weights, loss_form, rank_discount_form=None):
  """Returns the pairwise loss given the loss form.

  Args:
    labels: A list of graded relevance.
    scores: A list of item ranking scores.
    weights: A list of item weights.
    loss_form: A string representing the form of loss.
    rank_discount_form: A string representing the form of rank discount.

  Returns:
    A tuple of (sum of loss, sum of weights, count of nonzero weights).
  """
  scores, labels, weights = zip(
      *sorted(zip(scores, labels, weights), reverse=True))

  def _rank_discount(rank_discount_form, rank):
    discount = {
        'LINEAR': 1. / rank,
        'LOG': 1. / math.log(1. + rank),
    }
    return discount[rank_discount_form]

  def _lambda_weight(label_diff, i, j):
    delta = math.fabs(label_diff)
    if rank_discount_form is not None:
      delta *= math.fabs(
          _rank_discount(rank_discount_form, i + 1) -
          _rank_discount(rank_discount_form, j + 1))
    else:
      delta = 1. if delta > 0 else 0
    return delta

  def _loss(score_diff, label_diff, delta):
    if label_diff <= 0:
      return 0.
    loss_table = {
        ranking_losses.RankingLossKey.PAIRWISE_HINGE_LOSS:
            max(0, 1 - score_diff) * delta,
        ranking_losses.RankingLossKey.PAIRWISE_LOGISTIC_LOSS:
            math.log(1. + math.exp(-score_diff)) * delta,
        ranking_losses.RankingLossKey.PAIRWISE_SOFT_ZERO_ONE_LOSS:
            1. / (1. + math.exp(score_diff)) * delta,
    }
    return loss_table[loss_form]

  loss = 0.
  weight = 0.
  count = 0.
  for i in range(len(labels)):
    for j in range(len(labels)):
      if labels[i] > labels[j]:
        delta = _lambda_weight(labels[i] - labels[j], i, j)
        part_loss = _loss(scores[i] - scores[j], labels[i] - labels[j], delta)
        if weights[i] > 0:
          loss += part_loss * weights[i]
          weight += delta * weights[i]
          #与Reduction定义不一致
          # if weight > 0:
          if part_loss*weight>0.:
            count += 1.
  return loss, weight, count


def _batch_aggregation(batch_loss_list, reduction=None):
  """Returns the aggregated loss."""
  loss_sum = 0.
  weight_sum = 0.
  for loss, weight, count in batch_loss_list:
    loss_sum += loss
    if reduction == 'mean':
      weight_sum += weight
    else:
      weight_sum += count
  if weight_sum>0:
      return loss_sum / weight_sum
  else:
      return loss_sum


def _softmax(values):
  """Returns the softmax of `values`."""
  total = sum(math.exp(v) for v in values)
  return [math.exp(v) / total for v in values]


# Based on nn.sigmoid_cross_entropy_with_logits for x=logit and z=label the
# cross entropy is max(x, 0) - x * z + log(1 + exp(-abs(x)))
def _sigmoid_cross_entropy(labels, logits):

  def per_position_loss(logit, label):
    return max(logit, 0) - logit * label + math.log(1 + math.exp(-abs(logit)))

  return sum(
      per_position_loss(logit, label) for label, logit in zip(labels, logits))


# Aggregates the per position squared error.
def _mean_squared_error(logits, labels):
  return sum((logit - label)**2 for label, logit in zip(labels, logits))

class LossesTest(unittest.TestCase):

  def _check_pairwise_loss(self, loss_fn):
    """Helper function to test `loss_fn`."""
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 0., 2.]]
    listwise_weights = [[2.], [1.]]
    listwise_weights_expanded = [[2.] * 3, [1.] * 3]
    itemwise_weights = [[2., 3., 4.], [1., 1., 1.]]
    default_weights = [1.] * 3
    list_size = 3.
    loss_form_dict = {
          ranking_losses._pairwise_hinge_loss:
              ranking_losses.RankingLossKey.PAIRWISE_HINGE_LOSS,
          ranking_losses._pairwise_logistic_loss:
              ranking_losses.RankingLossKey.PAIRWISE_LOGISTIC_LOSS,
          ranking_losses._pairwise_soft_zero_one_loss:
              ranking_losses.RankingLossKey.PAIRWISE_SOFT_ZERO_ONE_LOSS,
      }
    loss_form = loss_form_dict[loss_fn]
    # Individual lists.
    self.assertAlmostEqual(
        loss_fn(torch.tensor([labels[0]]), torch.tensor([scores[0]])).item(),
        _batch_aggregation([
            _pairwise_loss(labels[0], scores[0], default_weights, loss_form)
        ]),
        places=5)
    self.assertAlmostEqual(
        loss_fn(torch.tensor([labels[1]]), torch.tensor([scores[1]])).item(),
        _batch_aggregation([
            _pairwise_loss(labels[1], scores[1], default_weights, loss_form)
        ]),
        places=5)

    # Itemwise weights.
    self.assertAlmostEqual(
        loss_fn(torch.tensor([labels[0]]), torch.tensor([scores[0]]),
                weights=torch.tensor([itemwise_weights[0]])).item(),
        _batch_aggregation([
            _pairwise_loss(labels[0], scores[0], itemwise_weights[0],
                           loss_form)
        ]),
        places=5)

    self.assertAlmostEqual(
        loss_fn(torch.tensor([labels[1]]), torch.tensor([scores[1]]),
                weights=torch.tensor([itemwise_weights[1]])).item(),
        _batch_aggregation([
            _pairwise_loss(labels[1], scores[1], itemwise_weights[1],
                           loss_form)
        ]),
        places=5)

    # Multiple lists.
    self.assertAlmostEqual(
        loss_fn(torch.tensor(labels), torch.tensor(scores), weights=torch.tensor(listwise_weights)).item(),
        _batch_aggregation([
            _pairwise_loss(labels[0], scores[0],
                           listwise_weights_expanded[0], loss_form),
            _pairwise_loss(labels[1], scores[1],
                           listwise_weights_expanded[1], loss_form)
        ]),
        places=5)

    # Test LambdaWeight.
    lambda_weight = losses_impl.DCGLambdaWeight(
        rank_discount_fn=lambda r: 1. / torch.log1p(r),
        smooth_fraction=1.)
    self.assertAlmostEqual(
        loss_fn(
            torch.tensor(labels),
            torch.tensor(scores),
            weights=torch.tensor(listwise_weights),
            lambda_weight=lambda_weight).item(),
        _batch_aggregation([
            _pairwise_loss(
                labels[0],
                scores[0],
                listwise_weights_expanded[0],
                loss_form,
                rank_discount_form='LOG'),
            _pairwise_loss(
                labels[1],
                scores[1],
                listwise_weights_expanded[1],
                loss_form,
                rank_discount_form='LOG')
        ]) * list_size,
        places=5)

    # Test loss reduction method.
    # Two reduction methods should return different loss values.
    reduced_1 = loss_fn(torch.tensor(labels), torch.tensor(scores), reduction=losses_impl.Reduction.SUM)
    reduced_2 = loss_fn(torch.tensor(labels), torch.tensor(scores),reduction=losses_impl.Reduction.MEAN)
    self.assertNotAlmostEqual(reduced_1.item(), reduced_2.item())

  def test_pairwise_hinge_loss(self):
    self._check_pairwise_loss(ranking_losses._pairwise_hinge_loss)

  def test_pairwise_logistic_loss(self):
    self._check_pairwise_loss(ranking_losses._pairwise_logistic_loss)

  def test_pairwise_soft_zero_one_loss(self):
    self._check_pairwise_loss(ranking_losses._pairwise_soft_zero_one_loss)

  def _check_make_pairwise_loss(self, loss_key):
    """Helper function to test `make_loss_fn`."""
    scores = [[1., 3., 2.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 0., 2.]]
    listwise_weights = [[2.], [1.]]
    listwise_weights_expanded = [[2.] * 3, [1.] * 3]
    itemwise_weights = [[2., 3., 4.], [1., 1., 1.]]
    default_weights = [1.] * 3
    weights_feature_name = 'weights'
    list_size = 3.
    features = {}

    loss_fn = ranking_losses.make_loss_fn(loss_key)
    # Individual lists.
    self.assertAlmostEqual(
        loss_fn(torch.tensor([labels[0]]), torch.tensor([scores[0]]), features).item(),
        _batch_aggregation([
            _pairwise_loss(labels[0], scores[0], default_weights, loss_key)
        ]),
        places=5)

    self.assertAlmostEqual(
        loss_fn(torch.tensor([labels[1]]), torch.tensor([scores[1]]), features).item(),
        _batch_aggregation([
            _pairwise_loss(labels[1], scores[1], default_weights, loss_key)
        ]),
        places=5)

        # Itemwise weights.
    loss_fn = ranking_losses.make_loss_fn(
        loss_key, weights_feature_name=weights_feature_name)
    features[weights_feature_name] = torch.tensor([itemwise_weights[0]])
    self.assertAlmostEqual(
        loss_fn(torch.tensor([labels[0]]), torch.tensor([scores[0]]), features).item(),
        _batch_aggregation([
            _pairwise_loss(labels[0], scores[0], itemwise_weights[0],
                           loss_key)
        ]),
        places=5)

    features[weights_feature_name] = torch.tensor([itemwise_weights[1]])
    self.assertAlmostEqual(
        loss_fn(torch.tensor([labels[1]]), torch.tensor([scores[1]]), features).item(),
        _batch_aggregation([
            _pairwise_loss(labels[1], scores[1], itemwise_weights[1],
                           loss_key)
        ]),
        places=5)

    # Multiple lists.
    features[weights_feature_name] = torch.tensor(listwise_weights)
    self.assertAlmostEqual(
        loss_fn(torch.tensor(labels), torch.tensor(scores), features).item(),
        _batch_aggregation([
            _pairwise_loss(labels[0], scores[0],
                           listwise_weights_expanded[0], loss_key),
            _pairwise_loss(labels[1], scores[1],
                           listwise_weights_expanded[1], loss_key)
        ]),
        places=5)

    # Test LambdaWeight.
    lambda_weight = losses_impl.DCGLambdaWeight(
        rank_discount_fn=lambda r: 1. / torch.log1p(r),
        smooth_fraction=1.)
    loss_fn = ranking_losses.make_loss_fn(
        loss_key,
        weights_feature_name=weights_feature_name,
        lambda_weight=lambda_weight)
    self.assertAlmostEqual(
        loss_fn(torch.tensor(labels), torch.tensor(scores), features).item(),
        _batch_aggregation([
            _pairwise_loss(
                labels[0],
                scores[0],
                listwise_weights_expanded[0],
                loss_key,
                rank_discount_form='LOG'),
            _pairwise_loss(
                labels[1],
                scores[1],
                listwise_weights_expanded[1],
                loss_key,
                rank_discount_form='LOG')
        ]) * list_size,
        places=5)

    # Test loss reduction method.
    # Two reduction methods should return different loss values.
    loss_fn_1 = ranking_losses.make_loss_fn(
        loss_key, reduction=losses_impl.Reduction.SUM)
    loss_fn_2 = ranking_losses.make_loss_fn(
        loss_key, reduction=losses_impl.Reduction.MEAN)
    self.assertNotAlmostEqual(
        loss_fn_1(torch.tensor(labels), torch.tensor(scores), features).item(),
        loss_fn_2(torch.tensor(labels), torch.tensor(scores), features).item())

  def test_make_pairwise_hinge_loss(self):
    self._check_make_pairwise_loss(
        ranking_losses.RankingLossKey.PAIRWISE_HINGE_LOSS)

  def test_make_pairwise_logistic_loss(self):
    self._check_make_pairwise_loss(
        ranking_losses.RankingLossKey.PAIRWISE_LOGISTIC_LOSS)

  def test_make_pairwise_soft_zero_one_loss(self):
    self._check_make_pairwise_loss(
        ranking_losses.RankingLossKey.PAIRWISE_SOFT_ZERO_ONE_LOSS)

  def test_softmax_loss(self):
    scores = [[1., 3., 2.], [1., 2., 3.], [1., 2., 3.]]
    labels = [[0., 0., 1.], [0., 0., 2.], [0., 0., 0.]]
    weights = [[2.], [1.], [1.]]
    self.assertAlmostEqual(
        ranking_losses._softmax_loss(torch.tensor(labels),torch.tensor(scores)).item(),
        -(math.log(_softmax(scores[0])[2]) +
          math.log(_softmax(scores[1])[2]) * 2.) / 2.,
        places=5)
    self.assertAlmostEqual(
        ranking_losses._softmax_loss(torch.tensor(labels),torch.tensor(scores), torch.tensor(weights)).item(),
        -(math.log(_softmax(scores[0])[2]) * 2. +
          math.log(_softmax(scores[1])[2]) * 2. * 1.) / 2.,
        places=5)
    # Test LambdaWeight.
    lambda_weight = losses_impl.DCGLambdaWeight(
        rank_discount_fn=lambda r: 1. / torch.log1p(r))
    self.assertAlmostEqual(
        ranking_losses._softmax_loss(
            torch.tensor(labels), torch.tensor(scores), lambda_weight=lambda_weight).item(),
        -(math.log(_softmax(scores[0])[2]) / math.log(1. + 2.) +
          math.log(_softmax(scores[1])[2]) * 2. / math.log(1. + 1.)) / 2.,
        places=5)

  def test_make_softmax_loss_fn(self):
      scores = [[1., 3., 2.], [1., 2., 3.]]
      labels = [[0., 0., 1.], [0., 0., 2.]]
      weights = [[2.], [1.]]
      weights_feature_name = 'weights'
      features = {weights_feature_name: torch.tensor(weights)}

      loss_fn_simple = ranking_losses.make_loss_fn(ranking_losses.RankingLossKey.SOFTMAX_LOSS)
      self.assertAlmostEqual(loss_fn_simple(torch.tensor(labels),torch.tensor(scores), features).item(),
                             -(math.log(_softmax(scores[0])[2]) +math.log(_softmax(scores[1])[2]) * 2.) / 2.,places=5)

      loss_fn_weighted = ranking_losses.make_loss_fn(ranking_losses.RankingLossKey.SOFTMAX_LOSS,
                            weights_feature_name=weights_feature_name)
      self.assertAlmostEqual(loss_fn_weighted(torch.tensor(labels),torch.tensor(scores), features).item(),
        -(math.log(_softmax(scores[0])[2]) * 2. +
          math.log(_softmax(scores[1])[2]) * 2. * 1.) / 2.,
        places=5)

    # Test loss reduction method.
    # Two reduction methods should return different loss values.
      loss_fn_1 = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.SOFTMAX_LOSS,
        reduction=losses_impl.Reduction.SUM)
      loss_fn_2 = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.SOFTMAX_LOSS,
        reduction=losses_impl.Reduction.MEAN)
      self.assertNotAlmostEqual(
        loss_fn_1(torch.tensor(labels),torch.tensor(scores), features).item(),
        loss_fn_2(torch.tensor(labels),torch.tensor(scores), features).item())

  def test_sigmoid_cross_entropy_loss(self):
    scores = [[0.2, 0.5, 0.3], [0.2, 0.3, 0.5], [0.2, 0.3, 0.5]]
    labels = [[0., 0., 1.], [0., 0., 2.], [0., 0., 0.]]
    weights = [[2.], [1.], [1.]]
    self.assertAlmostEqual(
        ranking_losses._sigmoid_cross_entropy_loss(torch.tensor(labels), torch.tensor(scores)).item(),
        (_sigmoid_cross_entropy(labels[0], scores[0]) +
         _sigmoid_cross_entropy(labels[1], scores[1]) +
         _sigmoid_cross_entropy(labels[2], scores[2])) / 9.,
        places=5)
    self.assertAlmostEqual(
        ranking_losses._sigmoid_cross_entropy_loss(torch.tensor(labels), torch.tensor(scores),
                                                   torch.tensor(weights)).item(),
        (_sigmoid_cross_entropy(labels[0], scores[0]) * 2.0 +
         _sigmoid_cross_entropy(labels[1], scores[1]) +
         _sigmoid_cross_entropy(labels[2], scores[2])) / 9.,
        places=5)

  def test_make_sigmoid_cross_entropy_loss_fn(self):
    scores = [[0.2, 0.5, 0.3], [0.2, 0.3, 0.5]]
    labels = [[0., 0., 1.], [0., 0., 1.]]
    weights = [[2.], [1.]]
    weights_feature_name = 'weights'
    features = {weights_feature_name: torch.tensor(weights)}
    loss_fn_simple = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.SIGMOID_CROSS_ENTROPY_LOSS)
    self.assertAlmostEqual(
        loss_fn_simple(torch.tensor(labels), torch.tensor(scores), features).item(),
        (_sigmoid_cross_entropy(labels[0], scores[0]) +
         _sigmoid_cross_entropy(labels[1], scores[1])) / 6.,
        places=5)
    loss_fn_weighted = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.SIGMOID_CROSS_ENTROPY_LOSS,
        weights_feature_name=weights_feature_name)
    self.assertAlmostEqual(
        loss_fn_weighted(torch.tensor(labels), torch.tensor(scores), features).item(),
        (_sigmoid_cross_entropy(labels[0], scores[0]) * 2.0 +
         _sigmoid_cross_entropy(labels[1], scores[1])) / 6.,
        places=5)

    # Test loss reduction method.
    # Two reduction methods should return different loss values.
    loss_fn_1 = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.SIGMOID_CROSS_ENTROPY_LOSS,
        reduction=losses_impl.Reduction.SUM)
    loss_fn_2 = ranking_losses.make_loss_fn(
        ranking_losses.RankingLossKey.SIGMOID_CROSS_ENTROPY_LOSS,
        reduction=losses_impl.Reduction.MEAN)
    self.assertNotAlmostEqual(
        loss_fn_1(torch.tensor(labels), torch.tensor(scores), features).item(),
        loss_fn_2(torch.tensor(labels), torch.tensor(scores), features).item())

  def test_mean_squared_loss(self):
      scores = [[0.2, 0.5, 0.3], [0.2, 0.3, 0.5], [0.2, 0.3, 0.5]]
      labels = [[0., 0., 1.], [0., 0., 2.], [0., 0., 0.]]
      weights = [[2.], [1.], [1.]]
      self.assertAlmostEqual(
            ranking_losses._mean_squared_loss(torch.tensor(labels), torch.tensor(scores)).item(),
            (_mean_squared_error(labels[0], scores[0]) +
             _mean_squared_error(labels[1], scores[1]) +
             _mean_squared_error(labels[2], scores[2])) / 9.,
            places=5)
      self.assertAlmostEqual(
            ranking_losses._mean_squared_loss(torch.tensor(labels), torch.tensor(scores), torch.tensor(weights)).item(),
            (_mean_squared_error(labels[0], scores[0]) * 2.0 +
             _mean_squared_error(labels[1], scores[1]) +
             _mean_squared_error(labels[2], scores[2])) / 9.,
            places=5)

  def test_make_mean_squared_loss_fn(self):
      scores = [[0.2, 0.5, 0.3], [0.2, 0.3, 0.5]]
      labels = [[0., 0., 1.], [0., 0., 1.]]
      weights = [[2.], [1.]]
      weights_feature_name = 'weights'
      features = {weights_feature_name: torch.tensor(weights)}
      loss_fn_simple = ranking_losses.make_loss_fn(
            ranking_losses.RankingLossKey.MEAN_SQUARED_LOSS)
      self.assertAlmostEqual(
            loss_fn_simple(torch.tensor(labels), torch.tensor(scores), features).item(),
            (_mean_squared_error(labels[0], scores[0]) +
             _mean_squared_error(labels[1], scores[1])) / 6.,
            places=5)

      loss_fn_weighted = ranking_losses.make_loss_fn(
            ranking_losses.RankingLossKey.MEAN_SQUARED_LOSS,
            weights_feature_name=weights_feature_name)
      self.assertAlmostEqual(
            loss_fn_weighted(torch.tensor(labels), torch.tensor(scores), features).item(),
            (_mean_squared_error(labels[0], scores[0]) * 2.0 +
             _mean_squared_error(labels[1], scores[1])) / 6.,
            places=5)

        # Test loss reduction method.
        # Two reduction methods should return different loss values.
      loss_fn_1 = ranking_losses.make_loss_fn(
            ranking_losses.RankingLossKey.MEAN_SQUARED_LOSS,
            reduction=losses_impl.Reduction.SUM)
      loss_fn_2 = ranking_losses.make_loss_fn(
            ranking_losses.RankingLossKey.MEAN_SQUARED_LOSS,
            reduction=losses_impl.Reduction.MEAN)
      self.assertNotAlmostEqual(
            loss_fn_1(torch.tensor(labels), torch.tensor(scores), features).item(),
          loss_fn_2(torch.tensor(labels), torch.tensor(scores), features).item())

  def test_list_mle_loss(self):
      scores = [[0., ln(3), ln(2)], [0., ln(2), ln(3)]]
      labels = [[0., 2., 1.], [1., 0., 2.]]
      weights = [[2.], [1.]]
      self.assertAlmostEqual(
            ranking_losses._list_mle_loss(torch.tensor(labels), torch.tensor(scores)).item(),
            -((ln(3. / (3 + 2 + 1)) + ln(2. / (2 + 1)) + ln(1. / 1)) +
              (ln(3. / (3 + 2 + 1)) + ln(1. / (1 + 2)) + ln(2. / 2))) / 2,
            places=5)
      self.assertAlmostEqual(
            ranking_losses._list_mle_loss(torch.tensor(labels), torch.tensor(scores), torch.tensor(weights)).item(),
            -(2 * (ln(3. / (3 + 2 + 1)) + ln(2. / (2 + 1)) + ln(1. / 1)) + 1 *
              (ln(3. / (3 + 2 + 1)) + ln(1. / (1 + 2)) + ln(2. / 2))) / 2,
            places=5)

  def test_list_mle_loss_tie(self):
      scores = [[0., ln(2), ln(3)]]
      labels = [[0., 0., 0.]]
      #[ln(3),ln(2),0]
      result1= -(ln(3. / (3 + 2 + 1)) + ln(2. / (2 + 1)) + ln(1. / 1))
      #[ln(2),0,ln(3)]
      result2= -(ln(2. / (3 + 2 + 1)) + ln(1. / (3 + 1)) + ln(3. / 3))
      #[0,ln(2),ln(3)]
      result3= -(ln(1/(3+2+1))+ln(2/(2+3))+ln(3./3.))
      #[0,ln(3),ln(2)]
      result4= -(ln(1/(3+2+1))+ln(3/(2+3))+ln(2./2.))
      #[ln(2),ln(3),0]
      result5=-(ln(2/(3+2+1))+ln(3/(1+3))+ln(1./1.))
      #[ln(3),0,ln(2)]
      result6=-(ln(3/(3+2+1))+ln(1/(2+1))+ln(2./2.))
      loss=ranking_losses._list_mle_loss(torch.tensor(labels), torch.tensor(scores)).item()
      self.assertIn(round(loss,5),
                    [round(r,5) for r in [result1,result2,result3,result4,result5,result6]])

  def test_list_mle_loss_lambda_weight(self):
      scores = [[0., ln(3), ln(2)], [0., ln(2), ln(3)]]
      labels = [[0., 2., 1.], [1., 0., 2.]]
      lw = ranking_losses.create_p_list_mle_lambda_weight(3)
      self.assertAlmostEqual(
            ranking_losses._list_mle_loss(torch.tensor(labels), torch.tensor(scores),
                                          lambda_weight=lw).item(),
            -((3 * ln(3. / (3 + 2 + 1)) + 1 * ln(2. /
                                                 (2 + 1)) + 0 * ln(1. / 1)) +
              (3 * ln(3. / (3 + 2 + 1)) + 1 * ln(1. /
                                                 (1 + 2)) + 0 * ln(2. / 2))) /
            2,
            places=5)

  def test_make_list_mle_loss_fn(self):
      scores = [[0., ln(3), ln(2)], [0., ln(2), ln(3)]]
      labels = [[0., 2., 1.], [1., 0., 2.]]
      weights = [[2.], [1.]]
      weights_feature_name = 'weights'
      features = {weights_feature_name: torch.tensor(weights)}
      loss_fn_simple = ranking_losses.make_loss_fn(
            ranking_losses.RankingLossKey.LIST_MLE_LOSS)
      self.assertAlmostEqual(
            loss_fn_simple(torch.tensor(labels), torch.tensor(scores), features).item(),
            -((ln(3. / (3 + 2 + 1)) + ln(2. / (2 + 1)) + ln(1. / 1)) +
              (ln(3. / (3 + 2 + 1)) + ln(1. / (1 + 2)) + ln(2. / 2))) / 2,
            places=5)
      loss_fn_weighted = ranking_losses.make_loss_fn(
            ranking_losses.RankingLossKey.LIST_MLE_LOSS,
            weights_feature_name=weights_feature_name)
      self.assertAlmostEqual(
            loss_fn_weighted(torch.tensor(labels), torch.tensor(scores), features).item(),
            -(2 * (ln(3. / (3 + 2 + 1)) + ln(2. / (2 + 1)) + ln(1. / 1)) + 1 *
              (ln(3. / (3 + 2 + 1)) + ln(1. / (1 + 2)) + ln(2. / 2))) / 2,
            places=5)

        # Test loss reduction method.
        # Two reduction methods should return different loss values.
      loss_fn_1 = ranking_losses.make_loss_fn(
            ranking_losses.RankingLossKey.LIST_MLE_LOSS,
            reduction=losses_impl.Reduction.SUM)
      loss_fn_2 = ranking_losses.make_loss_fn(
            ranking_losses.RankingLossKey.LIST_MLE_LOSS,
            reduction=losses_impl.Reduction.MEAN)
      self.assertNotAlmostEqual(
            loss_fn_1(torch.tensor(labels), torch.tensor(scores), features).item(),
            loss_fn_2(torch.tensor(labels), torch.tensor(scores), features).item())

  def test_approx_ndcg_loss(self):
      scores = [[1.4, -2.8, -0.4], [0., 1.8, 10.2], [1., 1.2, -3.2]]
      # ranks= [[1,    3,    2],   [3,  2,   1],    [2,  1,    3]]
      labels = [[0., 2., 1.], [1., 0., 3.], [0., 0., 0.]]
      weights = [[2.], [1.], [1.]]
      example_weights = [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]
      norm_wts = [sum([wt * l for wt, l in zip(wts, lbls)]) / sum(lbls)
                  if sum(lbls) else 0
                  for wts, lbls in zip(example_weights, labels)]

      self.assertAlmostEqual(
            ranking_losses._approx_ndcg_loss(torch.tensor(labels), torch.tensor(scores)).item(),
            -((1 / (3 / ln(2) + 1 / ln(3))) * (3 / ln(4) + 1 / ln(3)) +
              (1 / (7 / ln(2) + 1 / ln(3))) * (7 / ln(2) + 1 / ln(4))),
            places=5)
      self.assertAlmostEqual(
            ranking_losses._approx_ndcg_loss(torch.tensor(labels), torch.tensor(scores), torch.tensor(weights)).item(),
            -(2 * (1 / (3 / ln(2) + 1 / ln(3))) * (3 / ln(4) + 1 / ln(3)) + 1 *
              (1 / (7 / ln(2) + 1 / ln(3))) * (7 / ln(2) + 1 / ln(4))),
            places=5)
      self.assertAlmostEqual(
            ranking_losses._approx_ndcg_loss(torch.tensor(labels), torch.tensor(scores),
                                             torch.tensor(example_weights)).item(),
            -(norm_wts[0] * (1 / (3 / ln(2) + 1 / ln(3))) *
              (3 / ln(4) + 1 / ln(3)) +
              norm_wts[1] * (1 / (7 / ln(2) + 1 / ln(3))) *
              (7 / ln(2) + 1 / ln(4))),
            places=5)

  def test_make_approx_ndcg_fn(self):
      scores = [[1.4, -2.8, -0.4], [0., 1.8, 10.2], [1., 1.2, -3.2]]
      labels = [[0., 2., 1.], [1., 0., 3.], [0., 0., 0.]]
      weights = [[2.], [1.], [1.]]
      weights_feature_name = 'weights'
      features = {weights_feature_name: torch.tensor(weights)}
      loss_fn_simple = ranking_losses.make_loss_fn(
            ranking_losses.RankingLossKey.APPROX_NDCG_LOSS,
            reduction=losses_impl.Reduction.SUM)
      self.assertAlmostEqual(
            loss_fn_simple(torch.tensor(labels), torch.tensor(scores), features).item(),
            -((1 / (3 / ln(2) + 1 / ln(3))) * (3 / ln(4) + 1 / ln(3)) +
              (1 / (7 / ln(2) + 1 / ln(3))) * (7 / ln(2) + 1 / ln(4))),
            places=5)

      loss_fn_weighted = ranking_losses.make_loss_fn(
            ranking_losses.RankingLossKey.APPROX_NDCG_LOSS,
            weights_feature_name=weights_feature_name,
            reduction=losses_impl.Reduction.SUM)
      self.assertAlmostEqual(
            loss_fn_weighted(torch.tensor(labels), torch.tensor(scores), features).item(),
            -(2 * (1 / (3 / ln(2) + 1 / ln(3))) * (3 / ln(4) + 1 / ln(3)) + 1 *
              (1 / (7 / ln(2) + 1 / ln(3))) * (7 / ln(2) + 1 / ln(4))),
            places=5)

        # Test different alphas.
      loss_fn_1 = ranking_losses.make_loss_fn(
            ranking_losses.RankingLossKey.APPROX_NDCG_LOSS,
            extra_args={'alpha': 0.1})
      loss_fn_2 = ranking_losses.make_loss_fn(
            ranking_losses.RankingLossKey.APPROX_NDCG_LOSS,
            extra_args={'alpha': 100.})
      self.assertNotAlmostEqual(
            loss_fn_1(torch.tensor(labels), torch.tensor(scores), features).item(),
            loss_fn_2(torch.tensor(labels), torch.tensor(scores), features).item())

        # Test loss reduction method.
        # Two reduction methods should return different loss values.
      loss_fn_1 = ranking_losses.make_loss_fn(
            ranking_losses.RankingLossKey.APPROX_NDCG_LOSS,
            reduction=losses_impl.Reduction.SUM)
      loss_fn_2 = ranking_losses.make_loss_fn(
            ranking_losses.RankingLossKey.APPROX_NDCG_LOSS,
            reduction=losses_impl.Reduction.MEAN)
      self.assertNotAlmostEqual(
            loss_fn_1(torch.tensor(labels), torch.tensor(scores), features).item(),
            loss_fn_2(torch.tensor(labels), torch.tensor(scores), features).item())

  def test_approx_mrr_loss(self):
    scores = torch.tensor([[1.4, -2.8, -0.4], [0., 1.8, 10.2], [1., 1.2, -3.2]])
    labels = torch.tensor([[0., 0., 1.], [1., 0., 1.], [0., 0., 0.]])
    weights = torch.tensor([[2.], [1.], [1.]])
    self.assertAlmostEqual(
        ranking_losses._approx_mrr_loss(labels, scores).item(),
        -((1 / 2.) + 1 / 2. * (1 / 3. + 1 / 1.)),
        places=5)
    self.assertAlmostEqual(
        ranking_losses._approx_mrr_loss(labels, scores, weights).item(),
        -(2 * 1 / 2. + 1 * 1 / 2. * (1 / 3. + 1 / 1.)),
        places=5)

  def test_make_approx_mrr_fn(self):
      scores = torch.tensor([[1.4, -2.8, -0.4], [0., 1.8, 10.2], [1., 1.2, -3.2]])
      labels = torch.tensor([[0., 0., 1.], [1., 0., 1.], [0., 0., 0.]])
      weights = torch.tensor([[2.], [1.], [1.]])
      weights_feature_name = 'weights'
      features = {weights_feature_name: weights}
      loss_fn_simple = ranking_losses.make_loss_fn(
            ranking_losses.RankingLossKey.APPROX_MRR_LOSS,
            reduction=losses_impl.Reduction.SUM)
      self.assertAlmostEqual(
            loss_fn_simple(labels, scores, features).item(),
            -((1 / 2.) + 1 / 2. * (1 / 3. + 1 / 1.)),
            places=5)

      loss_fn_weighted = ranking_losses.make_loss_fn(
            ranking_losses.RankingLossKey.APPROX_MRR_LOSS,
            weights_feature_name=weights_feature_name,
            reduction=losses_impl.Reduction.SUM)
      self.assertAlmostEqual(
            loss_fn_weighted(labels, scores, features).item(),
            -(2 * 1 / 2. + 1 * 1 / 2. * (1 / 3. + 1 / 1.)),
            places=5)

        # Test different alphas.
      loss_fn_1 = ranking_losses.make_loss_fn(
            ranking_losses.RankingLossKey.APPROX_MRR_LOSS,
            extra_args={'alpha': 0.1})
      loss_fn_2 = ranking_losses.make_loss_fn(
            ranking_losses.RankingLossKey.APPROX_MRR_LOSS,
            extra_args={'alpha': 100.})
      self.assertNotAlmostEqual(
            loss_fn_1(labels, scores, features).item(),
            loss_fn_2(labels, scores, features).item())

        # Test loss reduction method.
        # Two reduction methods should return different loss values.
      loss_fn_1 = ranking_losses.make_loss_fn(
            ranking_losses.RankingLossKey.APPROX_MRR_LOSS,
            reduction=losses_impl.Reduction.SUM)
      loss_fn_2 = ranking_losses.make_loss_fn(
            ranking_losses.RankingLossKey.APPROX_MRR_LOSS,
            reduction=losses_impl.Reduction.MEAN)
      self.assertNotAlmostEqual(
            loss_fn_1(labels, scores, features).item(),
            loss_fn_2(labels, scores, features).item())


#undo 3d weight  failed
  def test_make_loss_fn(self):
      scores = torch.tensor([[0.2, 0.5, 0.3], [0.2, 0.3, 0.5]])
      labels = torch.tensor([[0., 0., 1.], [0., 0., 1.]])
      weights = torch.tensor([[2.], [1.]])
      weights_1d = torch.tensor([2., 1.])
      weights_3d = torch.tensor([[[2.], [1.], [0.]], [[0.], [1.], [2.]]])
      weights_feature_name = 'weights'
      weights_1d_feature_name = 'weights_1d'
      weights_3d_feature_name = 'weights_3d'
      features = {
          weights_feature_name: weights,
          weights_1d_feature_name: weights_1d,
          weights_3d_feature_name: weights_3d
      }
      pairwise_hinge_loss = ranking_losses._pairwise_hinge_loss(
            labels, scores).item()
      pairwise_hinge_loss_weighted = ranking_losses._pairwise_hinge_loss(
            labels, scores, weights=weights).item()
      pairwise_hinge_loss_itemwise_weighted = (
            ranking_losses._pairwise_hinge_loss(
                labels, scores, weights=torch.squeeze(weights_3d)).item())
      mean_squared_loss = ranking_losses._mean_squared_loss(labels,
                                                              scores).item()
      mean_squared_loss_weighted = ranking_losses._mean_squared_loss(
            labels, scores, weights=weights).item()
      mean_squared_loss_itemwise_weighted = ranking_losses._mean_squared_loss(
            labels, scores, weights=torch.squeeze(weights_3d)).item()

      loss_keys = [
            ranking_losses.RankingLossKey.PAIRWISE_HINGE_LOSS,
            ranking_losses.RankingLossKey.MEAN_SQUARED_LOSS
        ]
      loss_fn_simple = ranking_losses.make_loss_fn(loss_keys)
      self.assertAlmostEqual(
            loss_fn_simple(labels, scores, features).item(),
            pairwise_hinge_loss + mean_squared_loss,
            places=5)

        # With 2-d list-wise weighted examples.
      loss_fn_weighted_example = ranking_losses.make_loss_fn(
            loss_keys, weights_feature_name=weights_feature_name)
      self.assertAlmostEqual(
            loss_fn_weighted_example(labels, scores, features).item(),
            pairwise_hinge_loss_weighted + mean_squared_loss_weighted,
            places=5)

        # With 1-d list-wise weighted examples.
      loss_fn_weighted_example = ranking_losses.make_loss_fn(
            loss_keys, weights_feature_name=weights_1d_feature_name)
      self.assertAlmostEqual(
            loss_fn_weighted_example(labels, scores, features).item(),
            pairwise_hinge_loss_weighted + mean_squared_loss_weighted,
            places=5)

        # With 3-d item-wise weighted examples.
      loss_fn_weighted_example = ranking_losses.make_loss_fn(
            loss_keys, weights_feature_name=weights_3d_feature_name)
      self.assertAlmostEqual(
            loss_fn_weighted_example(labels, scores, features).item(),
            pairwise_hinge_loss_itemwise_weighted +
            mean_squared_loss_itemwise_weighted,
            places=5)

        # With both weighted loss and weighted examples.
      loss_weights = [3., 2.]
      weighted_loss_fn_weighted_example = ranking_losses.make_loss_fn(
            loss_keys, loss_weights, weights_feature_name=weights_feature_name)
      self.assertAlmostEqual(
            weighted_loss_fn_weighted_example(labels, scores, features).item(),
            pairwise_hinge_loss_weighted * loss_weights[0] +
            mean_squared_loss_weighted * loss_weights[1],
            places=5)

        # Test loss reduction method.
        # Two reduction methods should return different loss values.
      loss_fn_1 = ranking_losses.make_loss_fn(
            loss_keys, reduction=losses_impl.Reduction.SUM)
      loss_fn_2 = ranking_losses.make_loss_fn(
            loss_keys, reduction=losses_impl.Reduction.MEAN)
      self.assertNotAlmostEqual(
            loss_fn_1(labels, scores, features).item(),
            loss_fn_2(labels, scores, features).item())

        # Test invalid inputs.
      self.assertRaisesRegexp(ValueError,r'loss_keys cannot be None or empty.',
                              ranking_losses.make_loss_fn,[])
      self.assertRaisesRegexp(ValueError,r'loss_keys cannot be None or empty.',
                              ranking_losses.make_loss_fn,'')
      self.assertRaisesRegexp( ValueError, r'loss_keys and loss_weights must have the same size.',
                               ranking_losses.make_loss_fn,loss_keys, [2.0])
      invalid_loss_fn = ranking_losses.make_loss_fn(['invalid_key'])
      self.assertRaisesRegexp(ValueError,r'Invalid loss_key: invalid_key.',
                              invalid_loss_fn,labels, scores, features)

  def test_pairwise_logistic_loss_with_invalid_labels(self):
      scores = torch.tensor([[1., 3., 2.]])
      labels = torch.tensor([[0., -1., 1.]])
      self.assertAlmostEqual(
            ranking_losses._pairwise_logistic_loss(labels, scores).item(),
            math.log(1 + math.exp(-1.)),
            places=5)

  def test_softmax_loss_with_invalid_labels(self):
      scores = torch.tensor([[1., 3., 2.]])
      labels = torch.tensor([[0., -1., 1.]])
      self.assertAlmostEqual(
            ranking_losses._softmax_loss(labels, scores).item(),
            -(math.log(_softmax([1, 2])[1])),
            places=5)

  def test_sigmoid_cross_entropy_loss_with_invalid_labels(self):
      scores = torch.tensor([[1., 3., 2.]])
      labels = torch.tensor([[0., -1., 1.]])
      self.assertAlmostEqual(
            ranking_losses._sigmoid_cross_entropy_loss(labels, scores).item(),
            (math.log(1. + math.exp(-2.)) + math.log(1. + math.exp(1))) / 2,
            places=5)

  def test_mean_squared_loss_with_invalid_labels(self):
      scores = torch.tensor([[1., 3., 2.]])
      labels = torch.tensor([[0., -1., 1.]])
      self.assertAlmostEqual(
            ranking_losses._mean_squared_loss(labels, scores).item(),
            (1. + 1.) / 2,
            places=5)

class LossMetricTest(unittest.TestCase):

  def test_make_loss_metric_fn(self):
      scores = torch.tensor([[1., 3., 2.], [1., 2., 3.]])
      labels = torch.tensor([[0., 0., 1.], [0., 0., 2.]])
      weights = torch.tensor([[2.], [1.]])
      weights_feature_name = 'weights'
      features = {
          weights_feature_name: weights,
      }
      m = ranking_losses.make_loss_metric_fn(
          ranking_losses.RankingLossKey.SOFTMAX_LOSS)
      m_w = ranking_losses.make_loss_metric_fn(
          ranking_losses.RankingLossKey.SOFTMAX_LOSS,
          weights_feature_name=weights_feature_name)
      #weights [[1,1,1],[2,2,2]]
      self.assertAlmostEqual(m(labels, scores,features).item(), -(math.log(_softmax(scores[0])[2]) +
                          math.log(_softmax(scores[1])[2]) * 2.) / 9.,places=5)
      #weights [[2,2,2],[2,2,2]]
      self.assertAlmostEqual(m_w(labels, scores,features).item(), -(math.log(_softmax(scores[0])[2]) * 2. +
                            math.log(_softmax(scores[1])[2]) * 2. * 1.) / 12.,places=5)

      # Value of loss metric is the same as loss with MEAN reduction.
      loss_fn_mean = ranking_losses.make_loss_fn(
            ranking_losses.RankingLossKey.SOFTMAX_LOSS,
            reduction=losses_impl.Reduction.MEAN)
      loss_mean = loss_fn_mean(labels, scores, features)
      self.assertAlmostEqual(m(labels, scores, features).item(), loss_mean.item(),places=5)


if __name__ == "__main__":
    unittest.main()
