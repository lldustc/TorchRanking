import math
import torch
import torch.nn.functional as F
import abc
import utils

def _check_tensor_shapes(tensors):
    """Checks the tensor shapes to be compatible."""
    for tensor in tensors:
        assert (tensor.dim() >= 2)


def _apply_pairwise_op(op, tensor):
    """Applies the op on tensor in the pairwise manner."""
    _check_tensor_shapes([tensor])
    return op(torch.unsqueeze(tensor, 2), torch.unsqueeze(tensor, 1))


def _get_valid_pairs_and_clean_labels(labels):
    """Returns a boolean Tensor for valid pairs and cleaned labels."""
    assert (labels.dim() >= 2)
    is_valid = utils.is_label_valid(labels)
    valid_pairs = _apply_pairwise_op(utils.logical_and, is_valid)
    labels = torch.where(is_valid, labels, torch.zeros_like(labels))
    return valid_pairs, labels


class _LambdaWeight(object):
    """Interface for ranking metric optimization.

    This class wraps weights used in the LambdaLoss framework for ranking metric
    optimization (https://ai.google/research/pubs/pub47258). Such an interface is
    to be instantiated by concrete lambda weight models. The instance is used
    together with standard loss such as logistic loss and softmax loss.
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def pair_weights(self, labels, ranks):
        """Returns the weight adjustment `Tensor` for example pairs.

        Args:
          labels: A dense `Tensor` of labels with shape [batch_size, list_size].
          ranks: A dense `Tensor` of ranks with the same shape as `labels` that are
            sorted by logits.

        Returns:
          A `Tensor` that can weight example pairs.
        """
        raise NotImplementedError('Calling an abstract method.')

    def individual_weights(self, labels, ranks):
        """Returns the weight `Tensor` for individual examples.

        Args:
          labels: A dense `Tensor` of labels with shape [batch_size, list_size].
          ranks: A dense `Tensor` of ranks with the same shape as `labels` that are
            sorted by logits.

        Returns:
          A `Tensor` that can weight individual examples.
        """
        del ranks
        return labels


class DCGLambdaWeight(_LambdaWeight):
    """LambdaWeight for Discounted Cumulative Gain metric."""

    def __init__(self,
                 topn=None,
                 gain_fn=lambda label: label,
                 rank_discount_fn=lambda rank: 1. / rank,
                 normalized=False,
                 smooth_fraction=0.):
        """Constructor.

        Ranks are 1-based, not 0-based. Given rank i and j, there are two types of
        pair weights:
        u = |rank_discount_fn(|i-j|) - rank_discount_fn(|i-j| + 1)|
        v = |rank_discount_fn(i) - rank_discount_fn(j)|
        where u is the newly introduced one in LambdaLoss paper
        (https://ai.google/research/pubs/pub47258) and v is the original one in the
        LambdaMART paper "From RankNet to LambdaRank to LambdaMART: An Overview".
        The final pair weight contribution of ranks is
        (1-smooth_fraction) * u + smooth_fraction * v.

        Args:
        topn: (int) The topn for the DCG metric.
        gain_fn: (function) Transforms labels.
        rank_discount_fn: (function) The rank discount function.
        normalized: (bool) If True, normalize weight by the max DCG.
        smooth_fraction: (float) parameter to control the contribution from
        LambdaMART."""
        self._topn = topn
        self._gain_fn = gain_fn
        self._rank_discount_fn = rank_discount_fn
        self._normalized = normalized
        assert 0. <= smooth_fraction and smooth_fraction <= 1., (
                'smooth_fraction %s should be in range [0, 1].' % smooth_fraction)
        self._smooth_fraction = smooth_fraction

    def pair_weights(self, labels, ranks):
        """See `_LambdaWeight`."""
        #         labels=torch.tensor(labels)
        #         ranks=torch.tensor(ranks)
        _check_tensor_shapes([labels, ranks])
        valid_pair, labels = _get_valid_pairs_and_clean_labels(labels)
        gain = self._gain_fn(labels)
        if self._normalized:
            gain *= utils.inverse_max_dcg(
                labels,
                gain_fn=self._gain_fn,
                rank_discount_fn=self._rank_discount_fn,
                topn=self._topn)
        pair_gain = _apply_pairwise_op(torch.sub, gain)
        pair_gain *= valid_pair.float()
        list_size = labels.shape[1]
        topn = self._topn or list_size

        def _discount_for_relative_rank_diff():
            """Rank-based discount in the LambdaLoss paper."""
            # The LambdaLoss is not well defined when topn is active and topn <
            # list_size. We cap the rank of examples to topn + 1 so that the rank
            # differene is capped to topn. This is just a convenient upperbound
            # when topn is active. We need to revisit this.
            capped_rank = torch.where(
                ranks > topn,
                torch.ones_like(ranks) * (topn + 1), ranks)
            rank_diff = torch.abs(_apply_pairwise_op(torch.sub, capped_rank)).float()
            pair_discount = torch.where(
                rank_diff > 0,
                torch.abs(
                    self._rank_discount_fn(rank_diff) -
                    self._rank_discount_fn(rank_diff + 1)),
                torch.zeros_like(rank_diff))
            return pair_discount

        def _discount_for_absolute_rank():
            """Standard discount in the LambdaMART paper."""
            # When the rank discount is (1 / rank) for example, the discount is
            # |1 / r_i - 1 / r_j|. When i or j > topn, the discount becomes 0.
            rank_discount = torch.where(
                torch.gt(ranks, topn),
                torch.zeros_like(ranks.float()),
                self._rank_discount_fn(ranks.float()))
            pair_discount = torch.abs(_apply_pairwise_op(torch.sub, rank_discount))
            return pair_discount

        u = _discount_for_relative_rank_diff()
        v = _discount_for_absolute_rank()
        pair_discount = (1. -
                         self._smooth_fraction) * u + self._smooth_fraction * v
        pair_weight = torch.abs(pair_gain) * pair_discount
        if self._topn is None:
            return pair_weight
        pair_mask = _apply_pairwise_op(utils.logical_or,
                                       ranks <= self._topn)
        return pair_weight * pair_mask.float()

    def individual_weights(self, labels, ranks):
        """See `_LambdaWeight`."""
        _check_tensor_shapes([labels, ranks])
        labels = torch.where(utils.is_label_valid(labels), labels, torch.zeros_like(labels))
        gain = self._gain_fn(labels)
        if self._normalized:
            gain *= utils.inverse_max_dcg(
                labels,
                gain_fn=self._gain_fn,
                rank_discount_fn=self._rank_discount_fn,
                topn=self._topn)
        rank_discount = self._rank_discount_fn(ranks.float())
        return gain * rank_discount


class PrecisionLambdaWeight(_LambdaWeight):
    """LambdaWeight for Precision metric."""

    def __init__(self,
                 topn,
                 positive_fn=lambda label: label >= 1.0):
        """Constructor.

        Args:
          topn: (int) The K in Precision@K metric.
          positive_fn: (function): A function on `Tensor` that output boolean True
            for positive examples. The rest are negative examples.
        """
        self._topn = topn
        self._positive_fn = positive_fn

    def pair_weights(self, labels, ranks):
        """See `_LambdaWeight`.

        The current implementation here is that for any pairs of documents i and j,
        we set the weight to be 1 if
          - i and j have different labels.
          - i <= topn and j > topn or i > topn and j <= topn.
        This is exactly the same as the original LambdaRank method. The weight is
        the gain of swapping a pair of documents.

        Args:
          labels: A dense `Tensor` of labels with shape [batch_size, list_size].
          ranks: A dense `Tensor` of ranks with the same shape as `labels` that are
            sorted by logits.

        Returns:
          A `Tensor` that can weight example pairs.
        """
        _check_tensor_shapes([labels, ranks])
        valid_pair, labels = _get_valid_pairs_and_clean_labels(labels)
        binary_labels = self._positive_fn(labels).float()
        label_diff = torch.abs(_apply_pairwise_op(torch.sub, binary_labels))
        label_diff *= valid_pair.float()
        # i <= topn and j > topn or i > topn and j <= topn, i.e., xor(i <= topn, j
        # <= topn).
        rank_mask = _apply_pairwise_op(utils.logical_xor,
                                       ranks <= self._topn)
        return label_diff * rank_mask.float()


class ListMLELambdaWeight(_LambdaWeight):
    """LambdaWeight for ListMLE cost function."""

    def __init__(self, rank_discount_fn):
        """Constructor.

        Ranks are 1-based, not 0-based.

        Args:
          rank_discount_fn: (function) The rank discount function.
        """
        self._rank_discount_fn = rank_discount_fn

    def pair_weights(self, labels, ranks):
        """See `_LambdaWeight`."""
        pass

    def individual_weights(self, labels, ranks):
        """See `_LambdaWeight`."""
        _check_tensor_shapes([labels, ranks])
        rank_discount = self._rank_discount_fn(ranks.float())
        return torch.ones_like(labels) * rank_discount


def _compute_ranks(logits, is_valid):
    """Computes ranks by sorting valid logits.

    Args:
    logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
      ranking score of the corresponding item.
    is_valid: A `Tensor` of the same shape as `logits` representing validity of
      each entry.

    Returns:
    The `ranks` Tensor.
    """
    _check_tensor_shapes([logits, is_valid])
    # Only sort entries with is_valid = True.
    scores = torch.where(
        is_valid, logits, -1e-6 * torch.ones_like(logits) + torch.min(logits, dim=1, keepdim=True)[0])
    return utils.sorted_ranks(scores)


def _pairwise_comparison(labels, logits):
    r"""Returns pairwise comparison `Tensor`s.

    Given a list of n items, the labels of graded relevance l_i and the logits
    s_i, we form n^2 pairs. For each pair, we have the following:

                        /
                        | 1   if l_i > l_j for valid l_i and l_j.
    * `pairwise_labels` = |
                        | 0   otherwise
                        \
    * `pairwise_logits` = s_i - s_j

    Args:
    labels: A `Tensor` with shape [batch_size, list_size].
    logits: A `Tensor` with shape [batch_size, list_size].

    Returns:
    A tuple of (pairwise_labels, pairwise_logits) with each having the shape
    [batch_size, list_size, list_size].
    """
    # Compute the difference for all pairs in a list. The output is a Tensor with
    # shape [batch_size, list_size, list_size] where the entry [-1, i, j] stores
    # the information for pair (i, j).
    pairwise_label_diff = _apply_pairwise_op(torch.sub, labels)
    pairwise_logits = _apply_pairwise_op(torch.sub, logits)
    # Only keep the case when l_i > l_j.
    pairwise_labels = torch.gt(pairwise_label_diff, 0).float()
    is_valid = utils.is_label_valid(labels)
    valid_pair = _apply_pairwise_op(utils.logical_and, is_valid)
    pairwise_labels *= valid_pair.float()
    return pairwise_labels, pairwise_logits


_EPSILON = 1e-10


class Reduction:
    #just losses*weights no reduction
    NONE = "NONE"
    #torch.sum(losses*weights)
    SUM = "SUM"
    #torch.sum(losses*weights)/torch.sum(weights)
    MEAN = "MEAN"
    #torch.sum(losses*weights)/torch.sum(weights>0)
    SUM_BY_NONZERO_WEIGHTS = "SUM_BY_NONZERO_WEIGHTS"

    all = ["NONE", "SUM", "MEAN", "SUM_BY_NONZERO_WEIGHTS"]


class _RankingLoss(object):
    """Interface for ranking loss."""

    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def name(self):
        """The loss name."""
        raise NotImplementedError('Calling an abstract method.')

    @abc.abstractmethod
    def compute_unreduced_loss(self, labels, logits):
        """Computes the unreduced loss.

        Args:
          labels: A `Tensor` of the same shape as `logits` representing graded
            relevance.
          logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
            ranking score of the corresponding item.

        Returns:
          A tuple(losses, loss_weights) that have the same shape.
        """
        raise NotImplementedError('Calling an abstract method.')

    def normalize_weights(self, labels, weights):
        """Normalizes weights needed for tf.estimator (not tf.keras).

        This is needed for `tf.estimator` given that the reduction may be
        `SUM_OVER_NONZERO_WEIGHTS`. This function is not needed after we migrate
        from the deprecated reduction to `SUM` or `SUM_OVER_BATCH_SIZE`.

        Args:
          labels: A `Tensor` of shape [batch_size, list_size] representing graded
            relevance.
          weights: A scalar, a `Tensor` with shape [batch_size, 1] for list-wise
            weights, or a `Tensor` with shape [batch_size, list_size] for item-wise
            weights.

        Returns:
          The normalized weights.
        """
        del labels
        return 1.0 if weights is None else weights

    def sum_reduction(self, losses, weights, reduction):
        weights = weights * torch.ones_like(losses)
        if reduction is None:
            return losses * weights
        elif reduction == Reduction.SUM:
            return torch.sum(losses * weights)
        elif reduction == Reduction.MEAN:
            weights_sum = torch.sum(weights)
            if weights_sum > 0.:
                return torch.sum(losses * weights) / torch.sum(weights)
            else:
                return torch.sum(losses * weights)
        elif reduction == Reduction.SUM_BY_NONZERO_WEIGHTS:
            weight_losses=losses*weights
            weight_no_zero_num = torch.sum(torch.abs(weight_losses) > 0.)
            if weight_no_zero_num > 0:
                return torch.sum(weight_losses) / weight_no_zero_num
            else:
                return torch.sum(weight_losses)
        else:
            raise ValueError('Invalid reduction: {}'.format(reduction))

    def compute(self, labels, logits, weights=None, reduction=None):
        """Computes the reduced loss for tf.estimator (not tf.keras).

        Note that this function is not compatible with keras.

        Args:
          labels: A `Tensor` of the same shape as `logits` representing graded
            relevance.
          logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
            ranking score of the corresponding item.
          weights: A scalar, a `Tensor` with shape [batch_size, 1] for list-wise
            weights, or a `Tensor` with shape [batch_size, list_size] for item-wise
            weights.
          reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to
            reduce training loss over batch.

        Returns:
          Reduced loss for training and eval.
        """
        losses, loss_weights = self.compute_unreduced_loss(labels, logits)
        norm_weights = self.normalize_weights(labels, weights)
        weights = norm_weights * loss_weights
        return self.sum_reduction(losses, weights, reduction)

    def eval_metric(self, labels, logits, weights):
        """Computes the eval metric for the loss in tf.estimator (not tf.keras).

        Note that this function is not compatible with keras.

        Args:
          labels: A `Tensor` of the same shape as `logits` representing graded
            relevance.
          logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
            ranking score of the corresponding item.
          weights: A scalar, a `Tensor` with shape [batch_size, 1] for list-wise
            weights, or a `Tensor` with shape [batch_size, list_size] for item-wise
            weights.

        Returns:
          A metric op.
        """
        losses, loss_weights = self.compute_unreduced_loss(labels, logits)
        weights = self.normalize_weights(labels, weights) * loss_weights
        return self.sum_reduction(losses, weights, Reduction.MEAN)


class _PairwiseLoss(_RankingLoss):
    """Interface for pairwise ranking loss."""

    __metaclass__ = abc.ABCMeta

    def __init__(self, name, lambda_weight=None, params=None):
        """Constructor.

        Args:
          name: A string used as the name for this loss.
          lambda_weight: A `_LambdaWeight` object.
          params: A dict for params used in loss computation.
        """
        self._name = name
        self._lambda_weight = lambda_weight
        self._params = params or {}

    @property
    def name(self):
        """The loss name."""
        return self._name

    @abc.abstractmethod
    def _pairwise_loss(self, pairwise_logits):
        """The loss of pairwise logits with l_i > l_j."""
        raise NotImplementedError('Calling an abstract method.')

    def compute_unreduced_loss(self, labels, logits):
        """See `_RankingLoss`."""
        is_valid = utils.is_label_valid(labels)
        ranks = _compute_ranks(logits, is_valid)
        pairwise_labels, pairwise_logits = _pairwise_comparison(labels, logits)
        pairwise_weights = pairwise_labels
        if self._lambda_weight is not None:
            pairwise_weights *= self._lambda_weight.pair_weights(labels, ranks)
            # For LambdaLoss with relative rank difference, the scale of loss becomes
            # much smaller when applying LambdaWeight. This affects the training can
            # make the optimal learning rate become much larger. We use a heuristic to
            # scale it up to the same magnitude as standard pairwise loss.
            pairwise_weights *= float(labels.shape[1])

        #         pairwise_weights = tf.stop_gradient(
        #             pairwise_weights, name='weights_stop_gradient')
        pairwise_weights = pairwise_weights.clone().detach()
        return self._pairwise_loss(pairwise_logits), pairwise_weights

    def normalize_weights(self, labels, weights):
        """See _RankingLoss."""
        # The `weights` is item-wise and is applied non-symmetrically to update
        # pairwise_weights as
        #   pairwise_weights(i, j) = w_i * pairwise_weights(i, j).
        # This effectively applies to all pairs with l_i > l_j. Note that it is
        # actually symmetric when `weights` are constant per list, i.e., listwise
        # weights.
        if weights is None:
            weights = 1.
        weights = torch.where(utils.is_label_valid(labels),
                              torch.ones_like(labels) * weights,
                              torch.zeros_like(labels))
        return torch.unsqueeze(weights, dim=2)


class PairwiseLogisticLoss(_PairwiseLoss):
    """Implements pairwise logistic loss."""

    def _pairwise_loss(self, pairwise_logits):
        """See `_PairwiseLoss`."""
        # The following is the same as log(1 + exp(-pairwise_logits)).
        ##stabler,torch.log1p() in (0,log2)???
        return F.relu(-pairwise_logits) + torch.log1p(
            torch.exp(-torch.abs(pairwise_logits)))


class PairwiseHingeLoss(_PairwiseLoss):
    """Implements pairwise hinge loss."""

    def _pairwise_loss(self, pairwise_logits):
        """See `_PairwiseLoss`."""
        return F.relu(1 - pairwise_logits)


class PairwiseSoftZeroOneLoss(_PairwiseLoss):
    """Implements pairwise hinge loss."""

    def _pairwise_loss(self, pairwise_logits):
        """See `_PairwiseLoss`."""
        return torch.where(torch.gt(pairwise_logits, 0),
                           1. - torch.sigmoid(pairwise_logits),
                           torch.sigmoid(-pairwise_logits))


class _ListwiseLoss(_RankingLoss):
    """Interface for listwise loss."""

    def __init__(self, name, lambda_weight=None, params=None):
        """Constructor.

        Args:
          name: A string used as the name for this loss.
          lambda_weight: A `_LambdaWeight` object.
          params: A dict for params used in loss computation.
        """
        self._name = name
        self._lambda_weight = lambda_weight
        self._params = params or {}

    @property
    def name(self):
        """The loss name."""
        return self._name

    def normalize_weights(self, labels, weights):
        """See `_RankingLoss`."""
        if weights is None:
            return 1.0
        else:
            sum_labels = torch.sum(labels.float(), dim=1, keepdim=True)
            sum_weight_labels = torch.sum(weights.float() * labels.float(), dim=1, keepdim=True)
            sum_labels_no_zeros = torch.where(sum_labels > _EPSILON, sum_labels, torch.ones_like(sum_labels) * _EPSILON)
            sum_weight_lables_no_zeros = torch.where(sum_weight_labels > _EPSILON, sum_weight_labels,
                                                     torch.ones_like(sum_weight_labels) * _EPSILON)
            return torch.div(
                sum_weight_lables_no_zeros,
                sum_labels_no_zeros)


class SoftmaxLoss(_ListwiseLoss):
    """Implements softmax loss."""

    def precompute(self, labels, logits, weights):
        """Precomputes Tensors for softmax cross entropy inputs."""
        is_valid = utils.is_label_valid(labels)
        ranks = _compute_ranks(logits, is_valid)
        # Reset the invalid labels to 0 and reset the invalid logits to a logit with
        # ~= 0 contribution in softmax.
        labels = torch.where(is_valid, labels, torch.zeros_like(labels))
        logits = torch.where(is_valid, logits, math.log(_EPSILON) * torch.ones_like(logits))
        if self._lambda_weight is not None and isinstance(self._lambda_weight,
                                                          DCGLambdaWeight):
            labels = self._lambda_weight.individual_weights(labels, ranks)
        if weights is not None:
            labels *= weights
        return labels, logits

    def _softmax_cross_entropy_with_logits(self, labels_for_softmax, logits_for_softmax):
        return -torch.log(torch.softmax(logits_for_softmax, dim=1)) * labels_for_softmax

    def compute_unreduced_loss(self, labels, logits):
        """See `_RankingLoss`."""
        label_sum = torch.sum(labels, dim=1, keepdim=True)
        # Padding for rows with label_sum = 0.
        nonzero_mask = torch.gt(label_sum, 0.0)
        padded_labels = torch.where(nonzero_mask, labels, _EPSILON * torch.ones_like(labels))
        padded_label_sum = torch.sum(padded_labels, dim=1, keepdim=True)
        labels_for_softmax = padded_labels / padded_label_sum
        logits_for_softmax = logits
        # Padded labels have 0 weights in label_sum.
        weights_for_softmax = label_sum
        #         losses = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(
        #             labels_for_softmax, logits_for_softmax)
        losses = self._softmax_cross_entropy_with_logits(labels_for_softmax, logits_for_softmax)
        return losses, weights_for_softmax

    def compute(self, labels, logits, weights=None, reduction=None):
        """See `_RankingLoss`."""
        labels, logits = self.precompute(labels, logits, weights)
        losses, weights = self.compute_unreduced_loss(labels, logits)
        return self.sum_reduction(losses,weights,reduction)

    # #         return tf.compat.v1.losses.compute_weighted_loss(
    # #             losses, weights, reduction=reduction)

    def eval_metric(self, labels, logits, weights):
        """See `_RankingLoss`."""
        labels, logits = self.precompute(labels, logits, weights)
        losses, weights = self.compute_unreduced_loss(labels, logits)
        return self.sum_reduction(losses,weights,reduction=Reduction.MEAN)


class _PointwiseLoss(_RankingLoss):
    """Interface for pointwise loss."""

    def __init__(self, name, params=None):
        """Constructor.

        Args:
          name: A string used as the name for this loss.
          params: A dict for params used in loss computation.
        """
        self._name = name
        self._params = params or {}

    @property
    def name(self):
        """The loss name."""
        return self._name

    def normalize_weights(self, labels, weights):
        """See _RankingLoss."""
        if weights is None:
            weights = 1.
        return torch.where(utils.is_label_valid(labels),
                           torch.ones_like(labels) * weights,
                           torch.zeros_like(labels))


class SigmoidCrossEntropyLoss(_PointwiseLoss):
    """Implements sigmoid cross entropy loss."""

    def _sigmoid_cross_entropy_with_logits(self, labels, logits):
        return F.relu(logits) - logits * labels + torch.log1p(torch.exp(-torch.abs(logits)))

    def compute_unreduced_loss(self, labels, logits):
        """See `_RankingLoss`."""
        labels = torch.where(utils.is_label_valid(labels),
                             labels,
                             torch.zeros_like(labels))
        losses = self._sigmoid_cross_entropy_with_logits(labels, logits)
        return losses, 1.


class MeanSquaredLoss(_PointwiseLoss):
    """Implements the means squared error loss."""

    def compute_unreduced_loss(self, labels, logits):
        """See `_RankingLoss`."""
        is_valid = utils.is_label_valid(labels)
        labels = torch.where(is_valid, labels.float(), torch.zeros_like(labels))
        logits = torch.where(is_valid, logits.float(), torch.zeros_like(logits))
        losses = (labels - logits) ** 2
        return losses, 1.


class ListMLELoss(_ListwiseLoss):
    """Implements ListMLE loss."""

    def _cumsum_reverse(self, tensor, dim):
        tensor = torch.flip(tensor, (dim,))
        sums = torch.cumsum(tensor, dim=dim)
        return torch.flip(sums, (dim,))

    def compute_unreduced_loss(self, labels, logits):
        """See `_RankingLoss`."""
        is_valid = utils.is_label_valid(labels)
        # Reset the invalid labels to 0 and reset the invalid logits to a logit with
        # ~= 0 contribution.
        labels = torch.where(is_valid, labels.float(), torch.zeros_like(labels))
        logits = torch.where(is_valid, logits, math.log(_EPSILON) * torch.ones_like(logits))
        labels_min, _ = torch.min(labels, dim=1, keepdim=True)
        scores = torch.where(is_valid,
                             labels,
                             labels_min - 1e-6 * torch.ones_like(labels))
        sorted_labels, sorted_logits = utils.sort_by_scores(scores, [labels, logits])
        raw_max, _ = torch.max(sorted_logits, dim=1, keepdim=True)
        sorted_logits = sorted_logits - raw_max
        sums = self._cumsum_reverse(torch.exp(sorted_logits), 1)
        sums = torch.log(sums) - sorted_logits
        if self._lambda_weight is not None and isinstance(self._lambda_weight,
                                                          ListMLELambdaWeight):
            batch_size, list_size = sorted_labels.shape
            sums *= self._lambda_weight.individual_weights(sorted_labels,
                                                           torch.unsqueeze(torch.arange(list_size) + 1, 0).repeat(
                                                               batch_size, 1))
        negative_log_likelihood = torch.sum(sums, dim=1, keepdim=True)
        return negative_log_likelihood, 1.


class ApproxNDCGLoss(_ListwiseLoss):
    """Implements ApproxNDCG loss."""

    def compute_unreduced_loss(self, labels, logits):
        """See `_RankingLoss`."""
        alpha = self._params.get('alpha', 10.0)
        is_valid = utils.is_label_valid(labels)
        labels = torch.where(is_valid, labels, torch.zeros_like(labels))
        logits_min, _ = torch.min(logits, dim=-1, keepdim=True)
        logits = torch.where(
            is_valid, logits, -1e3 * torch.ones_like(logits) +
                              logits_min)

        label_sum = torch.sum(labels, dim=1, keepdim=True)
        nonzero_mask = torch.gt(label_sum, 0.0)
        labels = torch.where(nonzero_mask, labels,
                             _EPSILON * torch.ones_like(labels))
        gains = torch.pow(2., labels.float()) - 1.
        ranks = utils.approx_ranks(logits, alpha=alpha)
        discounts = 1. / torch.log1p(ranks)
        dcg = torch.sum(gains * discounts, dim=-1, keepdim=True)
        cost = -dcg * utils.inverse_max_dcg(labels)
        return cost, nonzero_mask.float()


class ApproxMRRLoss(_ListwiseLoss):
    """Implements ApproxMRR loss."""

    def compute_unreduced_loss(self, labels, logits):
        """See `_RankingLoss`."""
        alpha = self._params.get('alpha', 10.0)
        is_valid = utils.is_label_valid(labels)
        labels = torch.where(is_valid, labels, torch.zeros_like(labels))
        logits_min, _ = torch.min(logits, dim=-1, keepdim=True)
        logits = torch.where(is_valid,
                             logits,
                             -1e3 * torch.ones_like(logits) + logits_min)

        label_sum = torch.sum(labels, dim=1, keepdim=True)

        nonzero_mask = torch.gt(label_sum, 0.0)
        labels = torch.where(nonzero_mask, labels,
                             _EPSILON * torch.ones_like(labels))

        rr = 1. / utils.approx_ranks(logits, alpha=alpha)
        rr = torch.sum(rr * labels, dim=-1, keepdim=True)
        mrr = rr / torch.sum(labels, dim=-1, keepdim=True)
        return -mrr, nonzero_mask.float()
