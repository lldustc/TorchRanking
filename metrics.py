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

"""Defines ranking metrics as TF ops.

The metrics here are meant to be used during the TF training. That is, a batch
of instances in the Tensor format are evaluated by ops. It works with listwise
Tensors only.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import utils
import metrics_impl


class RankingMetricKey(object):
    """Ranking metric key strings."""
    # Mean Receiprocal Rank. For binary relevance.
    MRR = 'mrr'

    # Average Relevance Position.
    ARP = 'arp'

    # Normalized Discounted Culmulative Gain.
    NDCG = 'ndcg'

    # Discounted Culmulative Gain.
    DCG = 'dcg'

    # Precision. For binary relevance.
    PRECISION = 'precision'

    # Mean Average Precision. For binary relevance.
    MAP = 'map'

    # Ordered Pair Accuracy.
    ORDERED_PAIR_ACCURACY = 'ordered_pair_accuracy'


def make_ranking_metric_fn(
        metric_key,
        weights_feature_name=None,
        topn=None,
        name=None,
        gain_fn=metrics_impl.DEFAULT_GAIN_FN,
        rank_discount_fn=metrics_impl.DEFAULT_RANK_DISCOUNT_FN):
    """Factory method to create a ranking metric function.

    Args:
      metric_key: A key in `RankingMetricKey`.
      weights_feature_name: A `string` specifying the name of the weights feature
        in `features` dict.
      topn: An `integer` specifying the cutoff of how many items are considered in
        the metric.
      name: A `string` used as the name for this metric.
      gain_fn: (function) Transforms labels. A method to calculate gain
        parameters used in the definitions of the DCG and NDCG metrics, where the
        input is the relevance label of the item. The gain is often defined to be
        of the form 2^label-1.
      rank_discount_fn: (function) The rank discount function. A method to define
        the dicount parameters used in the definitions of DCG and NDCG metrics,
        where the input in the rank of item. The discount function is commonly
        defined to be of the form log(rank+1).

    Returns:
      A metric fn with the following Args:
      * `labels`: A `Tensor` of the same shape as `predictions` representing
      graded relevance.
      * `predictions`: A `Tensor` with shape [batch_size, list_size]. Each value
      is the ranking score of the corresponding example.
      * `features`: A dict of `Tensor`s that contains all features.
    """

    def _get_weights(features):
        """Get weights tensor from features and reshape it to 2-D if necessary."""
        weights = None
        if weights_feature_name:
            weights = features[weights_feature_name]
            # Convert weights to a 2-D Tensor.
            weights = utils.reshape_to_2d(weights)
        return weights

    def _average_relevance_position_fn(labels, predictions, features):
        """Returns average relevance position as the metric."""
        return average_relevance_position(
            labels, predictions, weights=_get_weights(features), name=name)

    def _mean_reciprocal_rank_fn(labels, predictions, features):
        """Returns mean reciprocal rank as the metric."""
        return mean_reciprocal_rank(
            labels,
            predictions,
            weights=_get_weights(features),
            topn=topn,
            name=name)

    def _normalized_discounted_cumulative_gain_fn(labels, predictions, features):
        """Returns normalized discounted cumulative gain as the metric."""
        return normalized_discounted_cumulative_gain(
            labels,
            predictions,
            weights=_get_weights(features),
            topn=topn,
            name=name,
            gain_fn=gain_fn,
            rank_discount_fn=rank_discount_fn)

    def _discounted_cumulative_gain_fn(labels, predictions, features):
        """Returns discounted cumulative gain as the metric."""
        return discounted_cumulative_gain(
            labels,
            predictions,
            weights=_get_weights(features),
            topn=topn,
            name=name,
            gain_fn=gain_fn,
            rank_discount_fn=rank_discount_fn)

    def _precision_fn(labels, predictions, features):
        """Returns precision as the metric."""
        return precision(
            labels,
            predictions,
            weights=_get_weights(features),
            topn=topn,
            name=name)

    def _mean_average_precision_fn(labels, predictions, features):
        """Returns mean average precision as the metric."""
        return mean_average_precision(
            labels,
            predictions,
            weights=_get_weights(features),
            topn=topn,
            name=name)

    def _ordered_pair_accuracy_fn(labels, predictions, features):
        """Returns ordered pair accuracy as the metric."""
        return ordered_pair_accuracy(
            labels, predictions, weights=_get_weights(features), name=name)

    metric_fn_dict = {
        RankingMetricKey.ARP: _average_relevance_position_fn,
        RankingMetricKey.MRR: _mean_reciprocal_rank_fn,
        RankingMetricKey.NDCG: _normalized_discounted_cumulative_gain_fn,
        RankingMetricKey.DCG: _discounted_cumulative_gain_fn,
        RankingMetricKey.PRECISION: _precision_fn,
        RankingMetricKey.MAP: _mean_average_precision_fn,
        RankingMetricKey.ORDERED_PAIR_ACCURACY: _ordered_pair_accuracy_fn,
    }
    assert metric_key in metric_fn_dict, ('metric_key %s not supported.' %
                                          metric_key)
    return metric_fn_dict[metric_key]


def mean_reciprocal_rank(labels,
                         predictions,
                         weights=None,
                         topn=None,
                         name=None):
    """Computes mean reciprocal rank (MRR).

    Args:
      labels: A `Tensor` of the same shape as `predictions`. A value >= 1 means a
        relevant example.
      predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
        the ranking score of the corresponding example.
      weights: A `Tensor` of the same shape of predictions or [batch_size, 1]. The
        former case is per-example and the latter case is per-list.
      topn: An integer cutoff specifying how many examples to consider for this
        metric. If None, the whole list is considered.
      name: A string used as the name for this metric.

    Returns:
      A metric for the weighted mean reciprocal rank of the batch.
    """
    metric = metrics_impl.MRRMetric(name, topn)
    return metric.compute(labels, predictions, weights)


def average_relevance_position(labels, predictions, weights=None, name=None):
    """Computes average relevance position (ARP).

    This can also be named as average_relevance_rank, but this can be confusing
    with mean_reciprocal_rank in acronyms. This name is more distinguishing and
    has been used historically for binary relevance as average_click_position.

    Args:
      labels: A `Tensor` of the same shape as `predictions`.
      predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
        the ranking score of the corresponding example.
      weights: A `Tensor` of the same shape of predictions or [batch_size, 1]. The
        former case is per-example and the latter case is per-list.
      name: A string used as the name for this metric.

    Returns:
      A metric for the weighted average relevance position.
    """
    metric = metrics_impl.ARPMetric(name)
    return metric.compute(labels, predictions, weights)


def precision(labels, predictions, weights=None, topn=None, name=None):
    """Computes precision as weighted average of relevant examples.

    Args:
      labels: A `Tensor` of the same shape as `predictions`. A value >= 1 means a
        relevant example.
      predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
        the ranking score of the corresponding example.
      weights: A `Tensor` of the same shape of predictions or [batch_size, 1]. The
        former case is per-example and the latter case is per-list.
      topn: A cutoff for how many examples to consider for this metric.
      name: A string used as the name for this metric.

    Returns:
      A metric for the weighted precision of the batch.
    """
    metric = metrics_impl.PrecisionMetric(name, topn)
    return metric.compute(labels, predictions, weights)


def mean_average_precision(labels,
                           predictions,
                           weights=None,
                           topn=None,
                           name=None):
    """Computes mean average precision (MAP).

    The implementation of MAP is based on Equation (1.7) in the following:
    Liu, T-Y "Learning to Rank for Information Retrieval" found at
    https://www.nowpublishers.com/article/DownloadSummary/INR-016

    Args:
      labels: A `Tensor` of the same shape as `predictions`. A value >= 1 means a
        relevant example.
      predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
        the ranking score of the corresponding example.
      weights: A `Tensor` of the same shape of predictions or [batch_size, 1]. The
        former case is per-example and the latter case is per-list.
      topn: A cutoff for how many examples to consider for this metric.
      name: A string used as the name for this metric.

    Returns:
      A metric for the mean average precision.
    """
    metric = metrics_impl.MeanAveragePrecisionMetric(name, topn)
    return metric.compute(labels, predictions, weights)


def normalized_discounted_cumulative_gain(
        labels,
        predictions,
        weights=None,
        topn=None,
        name=None,
        gain_fn=metrics_impl.DEFAULT_GAIN_FN,
        rank_discount_fn=metrics_impl.DEFAULT_RANK_DISCOUNT_FN):
    """Computes normalized discounted cumulative gain (NDCG).

    Args:
      labels: A `Tensor` of the same shape as `predictions`.
      predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
        the ranking score of the corresponding example.
      weights: A `Tensor` of the same shape of predictions or [batch_size, 1]. The
        former case is per-example and the latter case is per-list.
      topn: A cutoff for how many examples to consider for this metric.
      name: A string used as the name for this metric.
      gain_fn: (function) Transforms labels.
      rank_discount_fn: (function) The rank discount function.

    Returns:
      A metric for the weighted normalized discounted cumulative gain of the
      batch.
    """
    metric = metrics_impl.NDCGMetric(name, topn, gain_fn, rank_discount_fn)
    return metric.compute(labels, predictions, weights)


def discounted_cumulative_gain(
        labels,
        predictions,
        weights=None,
        topn=None,
        name=None,
        gain_fn=metrics_impl.DEFAULT_GAIN_FN,
        rank_discount_fn=metrics_impl.DEFAULT_RANK_DISCOUNT_FN):
    """Computes discounted cumulative gain (DCG).

    Args:
      labels: A `Tensor` of the same shape as `predictions`.
      predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
        the ranking score of the corresponding example.
      weights: A `Tensor` of the same shape of predictions or [batch_size, 1]. The
        former case is per-example and the latter case is per-list.
      topn: A cutoff for how many examples to consider for this metric.
      name: A string used as the name for this metric.
      gain_fn: (function) Transforms labels.
      rank_discount_fn: (function) The rank discount function.

    Returns:
      A metric for the weighted discounted cumulative gain of the batch.
    """
    metric = metrics_impl.DCGMetric(name, topn, gain_fn, rank_discount_fn)
    return metric.compute(labels, predictions, weights)


def ordered_pair_accuracy(labels, predictions, weights=None, name=None):
    """Computes the percentage of correctedly ordered pair.

    For any pair of examples, we compare their orders determined by `labels` and
    `predictions`. They are correctly ordered if the two orders are compatible.
    That is, labels l_i > l_j and predictions s_i > s_j and the weight for this
    pair is the weight from the l_i.

    Args:
      labels: A `Tensor` of the same shape as `predictions`.
      predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
        the ranking score of the corresponding example.
      weights: A `Tensor` of the same shape of predictions or [batch_size, 1]. The
        former case is per-example and the latter case is per-list.
      name: A string used as the name for this metric.

    Returns:
      A metric for the accuracy or ordered pairs.
    """
    metric = metrics_impl.OPAMetric(name)
    return metric.compute(labels, predictions, weights)


def eval_metric(metric_fn, **kwargs):
    """A stand-alone method to evaluate metrics on ranked results.

    Note that this method requires for the arguments of the metric to called
    explicitly. So, the correct usage is of the following form:
      tfr.metrics.eval_metric(tfr.metrics.mean_reciprocal_rank,
                              labels=my_labels,
                              predictions=my_scores).
    Here is a simple example showing how to use this method:
      import tensorflow_ranking as tfr
      scores = [[1., 3., 2.], [1., 2., 3.]]
      labels = [[0., 0., 1.], [0., 1., 2.]]
      weights = [[1., 2., 3.], [4., 5., 6.]]
      tfr.metrics.eval_metric(
          metric_fn=tfr.metrics.mean_reciprocal_rank,
          labels=labels,
          predictions=scores,
          weights=weights)
    Args:
      metric_fn: (function) Metric definition. A metric appearing in
        the TF-Ranking metrics module, e.g. tfr.metrics.mean_reciprocal_rank
      **kwargs: A collection of argument values to be passed to the metric, e.g.
        labels and predictions. See `_RankingMetric` and the various metric
        definitions in tfr.metrics for the specifics.

    Returns:
      The evaluation of the metric on the input ranked lists.

    Raises:
      ValueError: One of the arguments required by the metric is not provided in
        the list of arguments included in kwargs.

    """
    metric_spec = inspect.getargspec(metric_fn)
    metric_args = metric_spec.args
    required_metric_args = (metric_args[:-len(metric_spec.defaults)])
    for arg in required_metric_args:
        if arg not in kwargs:
            raise ValueError('Metric %s requires argument %s.'
                             % (metric_fn.__name__, arg))
    args = {}
    for arg in kwargs:
        if arg not in metric_args:
            raise ValueError('Metric %s does not accept argument %s.'
                             % (metric_fn.__name__, arg))
        args[arg] = kwargs[arg]
    return metric_fn(**args)
