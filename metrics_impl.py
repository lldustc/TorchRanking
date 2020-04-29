import abc
import torch
import utils

DEFAULT_GAIN_FN = lambda label: torch.pow(2.0, label) - 1

DEFAULT_RANK_DISCOUNT_FN = lambda rank: torch.log(torch.tensor(2.)) / torch.log1p(rank)

def _per_example_weights_to_per_list_weights(weights, relevance):
    """Computes per list weight from per example weight.

    Args:
      weights:  The weights `Tensor` of shape [batch_size, list_size].
      relevance:  The relevance `Tensor` of shape [batch_size, list_size].

    Returns:
      The per list `Tensor` of shape [batch_size, 1]
    """
    relevance_sum = torch.sum(relevance, dim=1, keepdim=True)
    relevance_sum = torch.where(relevance_sum > 0, relevance_sum, torch.ones_like(relevance_sum) * 1e-10)
    per_list_weights = torch.div(
        torch.sum(weights * relevance, dim=1, keepdim=True),
        relevance_sum)
    return per_list_weights

def _discounted_cumulative_gain(
        labels,
        weights=None,
        gain_fn=DEFAULT_GAIN_FN,
        rank_discount_fn=DEFAULT_RANK_DISCOUNT_FN):
    """Computes discounted cumulative gain (DCG).

    DCG = SUM(gain_fn(label) / rank_discount_fn(rank)). Using the default values
    of the gain and discount functions, we get the following commonly used
    formula for DCG: SUM((2^label -1) / log(1+rank)).

    Args:
      labels: The relevance `Tensor` of shape [batch_size, list_size]. For the
        ideal ranking, the examples are sorted by relevance in reverse order.
      weights: A `Tensor` of the same shape as labels or [batch_size, 1]. The
        former case is per-example and the latter case is per-list.
      gain_fn: (function) Transforms labels.
      rank_discount_fn: (function) The rank discount function.
    Returns:
      A `Tensor` as the weighted discounted cumulative gain per-list. The
      tensor shape is [batch_size, 1].
    """
    list_size = labels.shape[1]
    position = torch.arange(1, list_size + 1).float()
    gain = gain_fn(labels.float())
    discount = rank_discount_fn(position)
    return torch.sum(weights * gain * discount, dim=1, keepdim=True)


def _per_list_precision(labels, predictions, weights, topn):
    """Computes the precision for each query in the batch.

    Args:
      labels: A `Tensor` of the same shape as `predictions`. A value >= 1 means a
        relevant example.
      predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
        the ranking score of the corresponding example.
      weights: A `Tensor` of the same shape of predictions or [batch_size, 1]. The
        former case is per-example and the latter case is per-list.
      topn: A cutoff for how many examples to consider for this metric.

    Returns:
      A `Tensor` of size [batch_size, 1] containing the percision of each query
      respectively.
    """
    sorted_labels, sorted_weights = utils.sort_by_scores(
        predictions, [labels, weights], topn=topn)
    # Relevance = 1.0 when labels >= 1.0.
    relevance = torch.ge(sorted_labels, 1.0).float()
    w_sum = torch.sum(
        torch.ones_like(relevance) * sorted_weights,
        dim=1,
        keepdim=True)
    w_sum = torch.where(w_sum > 0, w_sum, torch.ones_like(w_sum) * 1e-10)
    per_list_precision = torch.div(
        torch.sum(relevance * sorted_weights, dim=1, keepdim=True),
        w_sum)
    return per_list_precision


def _is_compatible(t1, t2):
    if t1.dim() == t2.dim() and t1.shape == t2.shape:
        return True
    else:
        return False


def _assert_is_compatible(t1, t2):
    if not _is_compatible(t1, t2):
        raise ValueError("tensors are not compatible.")


def _assert_has_rank(t, rank):
    if t.dim() != rank:
        raise ValueError("tensor rank is not compatible.")


def _prepare_and_validate_params(labels, predictions, weights=None, topn=None):
    """Prepares and validates the parameters.

    Args:
      labels: A `Tensor` of the same shape as `predictions`. A value >= 1 means a
        relevant example.
      predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
        the ranking score of the corresponding example.
      weights: A `Tensor` of the same shape of predictions or [batch_size, 1]. The
        former case is per-example and the latter case is per-list.
      topn: A cutoff for how many examples to consider for this metric.

    Returns:
      (labels, predictions, weights, topn) ready to be used for metric
      calculation.
    """
    weights = 1.0 if weights is None else weights
    example_weights = torch.ones_like(labels) * weights
    _assert_is_compatible(predictions, example_weights)
    _assert_is_compatible(predictions, labels)
    _assert_has_rank(predictions,2)

    if topn is None:
        topn = predictions.shape[1]

    # All labels should be >= 0. Invalid entries are reset.
    is_label_valid = utils.is_label_valid(labels)
    labels = torch.where(is_label_valid, labels, torch.zeros_like(labels))
    pred_min, _ = torch.min(predictions, dim=1, keepdim=True)
    predictions = torch.where(
        is_label_valid, predictions, -1e-6 * torch.ones_like(predictions) + pred_min)
    return labels, predictions, example_weights, topn


class RankingMetric(object):
    """Interface for ranking metrics."""

    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def name(self):
        """The metric name."""
        raise NotImplementedError('Calling an abstract method.')

    def mean(self, measures, weights):
        w_sum = torch.sum(weights)
        if w_sum > 0.:
            return torch.sum(measures * weights) / w_sum
        else:
            return torch.sum(measures * weights)

    def divide_no_nan(self, t1, t2):
        t2 = torch.where(t2 > 0., t2, torch.ones_like(t2) * 1e-10)
        return torch.div(t1, t2)

    @abc.abstractmethod
    def compute(self, labels, predictions, weights):
        """Computes the metric with the given inputs.

        Args:
          labels: A `Tensor` of the same shape as `predictions` representing
            relevance.
          predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
            the ranking score of the corresponding example.
          weights: A `Tensor` of the same shape of predictions or [batch_size, 1].
            The former case is per-example and the latter case is per-list.

        Returns:
          A tf metric.
        """
        raise NotImplementedError('Calling an abstract method.')


class MRRMetric(RankingMetric):
    """Implements mean reciprocal rank (MRR)."""

    def __init__(self, name, topn):
        """Constructor."""
        self._name = name
        self._topn = topn

    @property
    def name(self):
        """The metric name."""
        return self._name

    def compute(self, labels, predictions, weights):
        """See `_RankingMetric`."""
        labels, predictions, weights, topn = _prepare_and_validate_params(
            labels, predictions, weights, self._topn)
        sorted_labels, = utils.sort_by_scores(predictions, [labels], topn=topn)
        sorted_list_size = sorted_labels.shape[1]
        # Relevance = 1.0 when labels >= 1.0 to accommodate graded relevance.
        relevance = torch.ge(sorted_labels, 1.0).float()
        reciprocal_rank = 1.0 / torch.arange(1, sorted_list_size + 1).float()
        # MRR has a shape of [batch_size, 1].
        mrr, _ = torch.max(relevance * reciprocal_rank, dim=1, keepdim=True)
        per_list_weights = _per_example_weights_to_per_list_weights(
            weights=weights,
            relevance=torch.ge(labels, 1.0).float())
        return self.mean(mrr, per_list_weights)

class ARPMetric(RankingMetric):
    """Implements average relevance position (ARP)."""

    def __init__(self, name):
        """Constructor."""
        self._name = name

    @property
    def name(self):
        """The metric name."""
        return self._name

    def compute(self, labels, predictions, weights):
        """See `_RankingMetric`."""
        list_size = predictions.shape[1]
        labels, predictions, weights, topn = _prepare_and_validate_params(
            labels, predictions, weights, list_size)
        sorted_labels, sorted_weights = utils.sort_by_scores(
            predictions, [labels, weights], topn=topn)
        relevance = sorted_labels * sorted_weights
        position = torch.arange(1, topn + 1).float()
        # TODO: Consider to add a cap poistion topn + 1 when there is no
        # relevant examples.
        return self.mean(position * torch.ones_like(relevance),
                         relevance)

class PrecisionMetric(RankingMetric):
    """Implements precision@k (P@k)."""

    def __init__(self, name, topn):
        """Constructor."""
        self._name = name
        self._topn = topn

    @property
    def name(self):
        """The metric name."""
        return self._name

    def compute(self, labels, predictions, weights):
        """See `_RankingMetric`."""
        labels, predictions, weights, topn = _prepare_and_validate_params(
            labels, predictions, weights, self._topn)
        per_list_precision = _per_list_precision(labels, predictions, weights, topn)
        # per_list_weights are computed from the whole list to avoid the problem of
        # 0 when there is no relevant example in topn.
        per_list_weights = _per_example_weights_to_per_list_weights(
            weights, torch.ge(labels, 1.0).float())
        return self.mean(per_list_precision, per_list_weights)

class MeanAveragePrecisionMetric(RankingMetric):
    """Implements mean average precision (MAP)."""

    def __init__(self, name, topn):
        """Constructor."""
        self._name = name
        self._topn = topn

    @property
    def name(self):
        """The metric name."""
        return self._name

    def compute(self, labels, predictions, weights):
        """See `_RankingMetric`."""
        labels, predictions, weights, topn = _prepare_and_validate_params(
            labels, predictions, weights, self._topn)
        sorted_labels, sorted_weights = utils.sort_by_scores(
            predictions, [labels, weights], topn=topn)
        # Relevance = 1.0 when labels >= 1.0.
        sorted_relevance = torch.ge(sorted_labels, 1.0).float()
        per_list_relevant_counts = torch.cumsum(sorted_relevance, dim=1)
        per_list_cutoffs = torch.cumsum(torch.ones_like(sorted_relevance), dim=1)
        # per_list_precisions = tf.math.divide_no_nan(per_list_relevant_counts,
        #                                             per_list_cutoffs)
        per_list_precisions = self.divide_no_nan(per_list_relevant_counts,
                                                 per_list_cutoffs)
        total_precision = torch.sum(
            per_list_precisions * sorted_weights * sorted_relevance,
            dim=1,
            keepdim=True)
        total_relevance = torch.sum(
            sorted_weights * sorted_relevance, dim=1, keepdim=True)
        # per_list_map = tf.math.divide_no_nan(total_precision, total_relevance)
        per_list_map = self.divide_no_nan(total_precision, total_relevance)
        # per_list_weights are computed from the whole list to avoid the problem of
        # 0 when there is no relevant example in topn.
        per_list_weights = _per_example_weights_to_per_list_weights(
            weights, torch.ge(labels, 1.0).float())
        return self.mean(per_list_map, per_list_weights)

class NDCGMetric(RankingMetric):
    """Implements normalized discounted cumulative gain (NDCG)."""

    def __init__(
            self,
            name,
            topn,
            gain_fn=DEFAULT_GAIN_FN,
            rank_discount_fn=DEFAULT_RANK_DISCOUNT_FN):
        """Constructor."""
        self._name = name
        self._topn = topn
        self._gain_fn = gain_fn
        self._rank_discount_fn = rank_discount_fn

    @property
    def name(self):
        """The metric name."""
        return self._name

    def compute(self, labels, predictions, weights):
        """See `_RankingMetric`."""
        labels, predictions, weights, topn = _prepare_and_validate_params(
            labels, predictions, weights, self._topn)
        sorted_labels, sorted_weights = utils.sort_by_scores(
            predictions, [labels, weights], topn=topn)
        dcg = _discounted_cumulative_gain(sorted_labels,
                                          sorted_weights,
                                          self._gain_fn,
                                          self._rank_discount_fn)
        # Sorting over the weighted labels to get ideal ranking.
        ideal_sorted_labels, ideal_sorted_weights = utils.sort_by_scores(
            weights * labels, [labels, weights], topn=topn)
        ideal_dcg = _discounted_cumulative_gain(ideal_sorted_labels,
                                                ideal_sorted_weights,
                                                self._gain_fn,
                                                self._rank_discount_fn)
        # per_list_ndcg = tf.compat.v1.math.divide_no_nan(dcg, ideal_dcg)
        per_list_ndcg = self.divide_no_nan(dcg, ideal_dcg)
        per_list_weights = _per_example_weights_to_per_list_weights(
            weights=weights,
            relevance=self._gain_fn(labels.float()))
        return self.mean(per_list_ndcg, per_list_weights)

class DCGMetric(RankingMetric):
    """Implements discounted cumulative gain (DCG)."""

    def __init__(
            self,
            name,
            topn,
            gain_fn=DEFAULT_GAIN_FN,
            rank_discount_fn=DEFAULT_RANK_DISCOUNT_FN):
        """Constructor."""
        self._name = name
        self._topn = topn
        self._gain_fn = gain_fn
        self._rank_discount_fn = rank_discount_fn

    @property
    def name(self):
        """The metric name."""
        return self._name

    def compute(self, labels, predictions, weights):
        """See `_RankingMetric`."""
        labels, predictions, weights, topn = _prepare_and_validate_params(
            labels, predictions, weights, self._topn)
        sorted_labels, sorted_weights = utils.sort_by_scores(
            predictions, [labels, weights], topn=topn)
        dcg = _discounted_cumulative_gain(sorted_labels,
                                          sorted_weights,
                                          self._gain_fn,
                                          self._rank_discount_fn)
        per_list_weights = _per_example_weights_to_per_list_weights(
            weights=weights,
            relevance=self._gain_fn(labels.float()))
        return self.mean(
            # tf.compat.v1.math.divide_no_nan(dcg, per_list_weights),
            torch.div(dcg, per_list_weights),
            per_list_weights)

class OPAMetric(RankingMetric):
    """Implements ordered pair accuracy (OPA)."""

    def __init__(self, name):
        """Constructor."""
        self._name = name

    @property
    def name(self):
        """The metric name."""
        return self._name

    def compute(self, labels, predictions, weights):
        """See `_RankingMetric`."""
        clean_labels, predictions, weights, _ = _prepare_and_validate_params(
            labels, predictions, weights)
        label_valid = torch.eq(clean_labels, labels)
        valid_pair = utils.logical_and(
            torch.unsqueeze(label_valid, 2), torch.unsqueeze(label_valid, 1))
        pair_label_diff = torch.unsqueeze(clean_labels, 2) - torch.unsqueeze(
            clean_labels, 1)
        pair_pred_diff = torch.unsqueeze(predictions, 2) - torch.unsqueeze(
            predictions, 1)
        # Correct pairs are represented twice in the above pair difference tensors.
        # We only take one copy for each pair.
        correct_pairs = torch.gt(pair_label_diff, 0).float() * torch.gt(pair_pred_diff, 0).float()
        pair_weights = torch.gt(pair_label_diff, 0).float() * torch.unsqueeze(weights, 2) * valid_pair.float()
        return self.mean(correct_pairs, pair_weights)