import torch
import losses_impl
import utils
class RankingLossKey(object):
    """Ranking loss key strings."""
    # Names for the ranking based loss functions.
    PAIRWISE_HINGE_LOSS = 'pairwise_hinge_loss'
    PAIRWISE_LOGISTIC_LOSS = 'pairwise_logistic_loss'
    PAIRWISE_SOFT_ZERO_ONE_LOSS = 'pairwise_soft_zero_one_loss'
    SOFTMAX_LOSS = 'softmax_loss'
    SIGMOID_CROSS_ENTROPY_LOSS = 'sigmoid_cross_entropy_loss'
    MEAN_SQUARED_LOSS = 'mean_squared_loss'
    LIST_MLE_LOSS = 'list_mle_loss'
    APPROX_NDCG_LOSS = 'approx_ndcg_loss'
    APPROX_MRR_LOSS = 'approx_mrr_loss'


def make_loss_fn(loss_keys,
                 loss_weights=None,
                 weights_feature_name=None,
                 lambda_weight=None,
                 reduction=losses_impl.Reduction.SUM_BY_NONZERO_WEIGHTS,
                 name=None,
                 extra_args=None):
    """Makes a loss function using a single loss or multiple losses.

    Args:
    loss_keys: A string or list of strings representing loss keys defined in
      `RankingLossKey`. Listed loss functions will be combined in a weighted
      manner, with weights specified by `loss_weights`. If `loss_weights` is
      None, default weight of 1 will be used.
    loss_weights: List of weights, same length as `loss_keys`. Used when merging
      losses to calculate the weighted sum of losses. If `None`, all losses are
      weighted equally with weight being 1.
    weights_feature_name: A string specifying the name of the weights feature in
      `features` dict.
    lambda_weight: A `_LambdaWeight` object created by factory methods like
      `create_ndcg_lambda_weight()`.
    reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to
      reduce training loss over batch.
    name: A string used as the name for this loss.
    extra_args: A string-keyed dictionary that contains any other loss-specific
      arguments.

    Returns:
        A function _loss_fn(). See `_loss_fn()` for its signature.

    Raises:
        ValueError: If `reduction` is invalid.
        ValueError: If `loss_keys` is None or empty.
        ValueError: If `loss_keys` and `loss_weights` have different sizes.
    """
    if reduction is not None and reduction not in losses_impl.Reduction.all:
        raise ValueError('Invalid reduction: {}'.format(reduction))

    if not loss_keys:
        raise ValueError('loss_keys cannot be None or empty.')

    if loss_weights:
        if len(loss_keys) != len(loss_weights):
            raise ValueError('loss_keys and loss_weights must have the same size.')

    if not isinstance(loss_keys, list):
        loss_keys = [loss_keys]

    def _loss_fn(labels, logits, features):
        """Computes a single loss or weighted combination of losses.

        Args:
          labels: A `Tensor` of the same shape as `logits` representing relevance.
          logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
            ranking score of the corresponding item.
          features: Dict of Tensors of shape [batch_size, list_size, ...] for
            per-example features and shape [batch_size, ...] for non-example context
            features.

        Returns:
          An op for a single loss or weighted combination of multiple losses.

        Raises:
          ValueError: If `loss_keys` is invalid.
        """
        weights = None
        if weights_feature_name:
            weights = features[weights_feature_name]
            # Convert weights to a 2-D Tensor.
            weights = utils.reshape_to_2d(weights)

        loss_kwargs = {
            'labels': labels,
            'logits': logits,
            'weights': weights,
            'reduction': reduction,
            'name': name,
        }
        if extra_args is not None:
            loss_kwargs.update(extra_args)
        loss_kwargs_with_lambda_weight = loss_kwargs.copy()
        loss_kwargs_with_lambda_weight['lambda_weight'] = lambda_weight
        key_to_fn = {
            RankingLossKey.PAIRWISE_HINGE_LOSS:
                (_pairwise_hinge_loss, loss_kwargs_with_lambda_weight),
            RankingLossKey.PAIRWISE_LOGISTIC_LOSS:
                (_pairwise_logistic_loss, loss_kwargs_with_lambda_weight),
            RankingLossKey.PAIRWISE_SOFT_ZERO_ONE_LOSS:
                (_pairwise_soft_zero_one_loss, loss_kwargs_with_lambda_weight),
            RankingLossKey.SOFTMAX_LOSS:
                (_softmax_loss, loss_kwargs_with_lambda_weight),
            RankingLossKey.SIGMOID_CROSS_ENTROPY_LOSS:
                (_sigmoid_cross_entropy_loss, loss_kwargs),
            RankingLossKey.MEAN_SQUARED_LOSS: (_mean_squared_loss, loss_kwargs),
            RankingLossKey.LIST_MLE_LOSS:
                (_list_mle_loss, loss_kwargs_with_lambda_weight),
            RankingLossKey.APPROX_NDCG_LOSS: (_approx_ndcg_loss, loss_kwargs),
            RankingLossKey.APPROX_MRR_LOSS: (_approx_mrr_loss, loss_kwargs),
        }

        # Obtain the list of loss ops.
        loss_ops = []
        for loss_key in loss_keys:
            if loss_key not in key_to_fn:
                raise ValueError('Invalid loss_key: {}.'.format(loss_key))
            loss_fn, kwargs = key_to_fn[loss_key]
            loss_ops.append(loss_fn(**kwargs))
        # Compute weighted combination of losses.
        if loss_weights is not None:
            weighted_losses = []
            for loss_op, loss_weight in zip(loss_ops, loss_weights):
                weighted_losses.append(loss_op * loss_weight)
        else:
            weighted_losses = loss_ops
        return sum(weighted_losses)

    return _loss_fn


def make_loss_metric_fn(loss_key,
                        weights_feature_name=None,
                        lambda_weight=None,
                        name=None):
    """Factory method to create a metric based on a loss.

    Args:
    loss_key: A key in `RankingLossKey`.
    weights_feature_name: A `string` specifying the name of the weights feature
      in `features` dict.
    lambda_weight: A `_LambdaWeight` object.
    name: A `string` used as the name for this metric.

    Returns:
    A metric fn with the following Args:
    * `labels`: A `Tensor` of the same shape as `predictions` representing
    graded relevance.
    * `predictions`: A `Tensor` with shape [batch_size, list_size]. Each value
    is the ranking score of the corresponding example.
    * `features`: A dict of `Tensor`s that contains all features.
    """

    metric_dict = {
        RankingLossKey.PAIRWISE_HINGE_LOSS:
            losses_impl.PairwiseHingeLoss(name, lambda_weight=lambda_weight),
        RankingLossKey.PAIRWISE_LOGISTIC_LOSS:
            losses_impl.PairwiseLogisticLoss(name, lambda_weight=lambda_weight),
        RankingLossKey.PAIRWISE_SOFT_ZERO_ONE_LOSS:
            losses_impl.PairwiseSoftZeroOneLoss(
                name, lambda_weight=lambda_weight),
        RankingLossKey.SOFTMAX_LOSS:
            losses_impl.SoftmaxLoss(name, lambda_weight=lambda_weight),
        RankingLossKey.SIGMOID_CROSS_ENTROPY_LOSS:
            losses_impl.SigmoidCrossEntropyLoss(name),
        RankingLossKey.MEAN_SQUARED_LOSS:
            losses_impl.MeanSquaredLoss(name),
        RankingLossKey.LIST_MLE_LOSS:
            losses_impl.ListMLELoss(name, lambda_weight=lambda_weight),
        RankingLossKey.APPROX_NDCG_LOSS:
            losses_impl.ApproxNDCGLoss(name),
        RankingLossKey.APPROX_MRR_LOSS:
            losses_impl.ApproxMRRLoss(name),
    }

    def _get_weights(features):
        """Get weights tensor from features and reshape it to 2-D if necessary."""
        weights = None
        if weights_feature_name:
            weights = features[weights_feature_name]
            # Convert weights to a 2-D Tensor.
            weights = utils.reshape_to_2d(weights)
        return weights

    def metric_fn(labels, predictions, features):
        """Defines the metric fn."""
        weights = _get_weights(features)
        loss = metric_dict.get(loss_key, None)
        if loss is None:
            raise ValueError('loss_key {} not supported.'.format(loss_key))
        return loss.eval_metric(labels, predictions, weights)

    return metric_fn


def create_ndcg_lambda_weight(topn=None, smooth_fraction=0.):
    """Creates _LambdaWeight for NDCG metric."""
    return losses_impl.DCGLambdaWeight(
        topn,
        gain_fn=lambda labels: torch.pow(2.0, labels) - 1.,
        rank_discount_fn=lambda rank: 1. / torch.log1p(rank),
        normalized=True,
        smooth_fraction=smooth_fraction)


def create_reciprocal_rank_lambda_weight(topn=None, smooth_fraction=0.):
    """Creates _LambdaWeight for MRR-like metric."""
    return losses_impl.DCGLambdaWeight(
        topn,
        gain_fn=lambda labels: labels,
        rank_discount_fn=lambda rank: 1. / rank,
        normalized=True,
        smooth_fraction=smooth_fraction)


def create_p_list_mle_lambda_weight(list_size):
    """Creates _LambdaWeight based on Position-Aware ListMLE paper.

    Produces a weight based on the formulation presented in the
    "Position-Aware ListMLE" paper (Lan et al.) and available using
    create_p_list_mle_lambda_weight() factory function above.

    Args:
    list_size: Size of the input list.

    Returns:
    A _LambdaWeight for Position-Aware ListMLE.
    """
    return losses_impl.ListMLELambdaWeight(
        rank_discount_fn=lambda rank: torch.pow(2., list_size - rank) - 1.)


def _pairwise_hinge_loss(
        labels,
        logits,
        weights=None,
        lambda_weight=None,
        reduction=losses_impl.Reduction.SUM_BY_NONZERO_WEIGHTS,
        name=None):
    """Computes the pairwise hinge loss for a list.

    The hinge loss is defined as Hinge(l_i > l_j) = max(0, 1 - (s_i - s_j)). So a
    correctly ordered pair has 0 loss if (s_i - s_j >= 1). Otherwise the loss
    increases linearly with s_i - s_j. When the list_size is 2, this reduces to
    the standard hinge loss.

    Args:
    labels: A `Tensor` of the same shape as `logits` representing graded
      relevance.
    logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
      ranking score of the corresponding item.
    weights: A scalar, a `Tensor` with shape [batch_size, 1] for list-wise
      weights, or a `Tensor` with shape [batch_size, list_size] for item-wise
      weights.
    lambda_weight: A `_LambdaWeight` object.
    reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to
      reduce training loss over batch.
    name: A string used as the name for this loss.

    Returns:
    An op for the pairwise hinge loss.
    """
    loss = losses_impl.PairwiseHingeLoss(name, lambda_weight)
    return loss.compute(labels, logits, weights, reduction)


def _pairwise_logistic_loss(
        labels,
        logits,
        weights=None,
        lambda_weight=None,
        reduction=losses_impl.Reduction.SUM_BY_NONZERO_WEIGHTS,
        name=None):
    """Computes the pairwise logistic loss for a list.

    The preference probability of each pair is computed as the sigmoid function:
    P(l_i > l_j) = 1 / (1 + exp(s_j - s_i)) and the logistic loss is log(P(l_i >
    l_j)) if l_i > l_j and 0 otherwise.

    Args:
    labels: A `Tensor` of the same shape as `logits` representing graded
      relevance.
    logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
      ranking score of the corresponding item.
    weights: A scalar, a `Tensor` with shape [batch_size, 1] for list-wise
      weights, or a `Tensor` with shape [batch_size, list_size] for item-wise
      weights.
    lambda_weight: A `_LambdaWeight` object.
    reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to
      reduce training loss over batch.
    name: A string used as the name for this loss.

    Returns:
    An op for the pairwise logistic loss.
    """
    loss = losses_impl.PairwiseLogisticLoss(name, lambda_weight)
    return loss.compute(labels, logits, weights, reduction)


def _pairwise_soft_zero_one_loss(
        labels,
        logits,
        weights=None,
        lambda_weight=None,
        reduction=losses_impl.Reduction.SUM_BY_NONZERO_WEIGHTS,
        name=None):
    """Computes the pairwise soft zero-one loss.

    Note this is different from sigmoid cross entropy in that soft zero-one loss
    is a smooth but non-convex approximation of zero-one loss. The preference
    probability of each pair is computed as the sigmoid function: P(l_i > l_j) = 1
    / (1 + exp(s_j - s_i)). Then 1 - P(l_i > l_j) is directly used as the loss.
    So a correctly ordered pair has a loss close to 0, while an incorrectly
    ordered pair has a loss bounded by 1.

    Args:
    labels: A `Tensor` of the same shape as `logits` representing graded
      relevance.
    logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
      ranking score of the corresponding item.
    weights: A scalar, a `Tensor` with shape [batch_size, 1] for list-wise
      weights, or a `Tensor` with shape [batch_size, list_size] for item-wise
      weights.
    lambda_weight: A `_LambdaWeight` object.
    reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to
      reduce training loss over batch.
    name: A string used as the name for this loss.

    Returns:
    An op for the pairwise soft zero one loss.
    """
    loss = losses_impl.PairwiseSoftZeroOneLoss(name, lambda_weight)
    return loss.compute(labels, logits, weights, reduction)


def _softmax_loss(
        labels,
        logits,
        weights=None,
        lambda_weight=None,
        reduction=losses_impl.Reduction.SUM_BY_NONZERO_WEIGHTS,
        name=None):
    """Computes the softmax cross entropy for a list.

    This is the ListNet loss originally proposed by Cao et al.
    ["Learning to Rank: From Pairwise Approach to Listwise Approach"] and is
    appropriate for datasets with binary relevance labels [see "An Analysis of
    the Softmax Cross Entropy Loss for Learning-to-Rank with Binary Relevance" by
    Bruch et al.]

    Given the labels l_i and the logits s_i, we sort the examples and obtain ranks
    r_i. The standard softmax loss doesn't need r_i and is defined as
      -sum_i l_i * log(exp(s_i) / (exp(s_1) + ... + exp(s_n))).
    The `lambda_weight` re-weight examples based on l_i and r_i.
      -sum_i w(l_i, r_i) * log(exp(s_i) / (exp(s_1) + ... + exp(s_n))).abc
    See 'individual_weights' in 'DCGLambdaWeight' for how w(l_i, r_i) is computed.

    Args:
    labels: A `Tensor` of the same shape as `logits` representing graded
      relevance.
    logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
      ranking score of the corresponding item.
    weights: A scalar, a `Tensor` with shape [batch_size, 1] for list-wise
      weights, or a `Tensor` with shape [batch_size, list_size] for item-wise
      weights.
    lambda_weight: A `DCGLambdaWeight` instance.
    reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to
      reduce training loss over batch.
    name: A string used as the name for this loss.

    Returns:
    An op for the softmax cross entropy as a loss.
    """
    loss = losses_impl.SoftmaxLoss(name, lambda_weight)
    return loss.compute(labels, logits, weights, reduction)


def _sigmoid_cross_entropy_loss(
        labels,
        logits,
        weights=None,
        reduction=losses_impl.Reduction.SUM_BY_NONZERO_WEIGHTS,
        name=None):
    """Computes the sigmoid_cross_entropy loss for a list.

    Given the labels of graded relevance l_i and the logits s_i, we calculate
    the sigmoid cross entropy for each ith position and aggregate the per position
    losses.

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
    name: A string used as the name for this loss.

    Returns:
    An op for the sigmoid cross entropy as a loss.
    """
    loss = losses_impl.SigmoidCrossEntropyLoss(name)
    return loss.compute(labels, logits, weights, reduction)


def _mean_squared_loss(
        labels,
        logits,
        weights=None,
        reduction=losses_impl.Reduction.SUM_BY_NONZERO_WEIGHTS,
        name=None):
    """Computes the mean squared loss for a list.

    Given the labels of graded relevance l_i and the logits s_i, we calculate
    the squared error for each ith position and aggregate the per position
    losses.

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
    name: A string used as the name for this loss.

    Returns:
    An op for the mean squared error as a loss.
    """
    loss = losses_impl.MeanSquaredLoss(name)
    return loss.compute(labels, logits, weights, reduction)


def _list_mle_loss(
        labels,
        logits,
        weights=None,
        lambda_weight=None,
        reduction=losses_impl.Reduction.SUM_BY_NONZERO_WEIGHTS,
        name=None):
    """Computes the ListMLE loss [Xia et al.

    2008] for a list.

    Given the labels of graded relevance l_i and the logits s_i, we calculate
    the ListMLE loss for the given list.

    The `lambda_weight` re-weights examples based on l_i and r_i.
    The recommended weighting scheme is the formulation presented in the
    "Position-Aware ListMLE" paper (Lan et al.) and available using
    create_p_list_mle_lambda_weight() factory function above.

    Args:
    labels: A `Tensor` of the same shape as `logits` representing graded
      relevance.
    logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
      ranking score of the corresponding item.
    weights: A scalar, a `Tensor` with shape [batch_size, 1] for list-wise
      weights, or a `Tensor` with shape [batch_size, list_size] for item-wise
      weights.
    lambda_weight: A `DCGLambdaWeight` instance.
    reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to
      reduce training loss over batch.
    name: A string used as the name for this loss.

    Returns:
    An op for the ListMLE loss.
    """
    loss = losses_impl.ListMLELoss(name, lambda_weight)
    return loss.compute(labels, logits, weights, reduction)


def _approx_ndcg_loss(labels,
                      logits,
                      weights=None,
                      reduction=losses_impl.Reduction.SUM,
                      name=None,
                      alpha=10.):
    """Computes ApproxNDCG loss.

    ApproxNDCG ["A general approximation framework for direct optimization of
    information retrieval measures" by Qin et al.] is a smooth approximation
    to NDCG. Its performance on datasets with graded relevance is competitive
    with other state-of-the-art algorithms [see "Revisiting Approximate Metric
    Optimization in the Age of Deep Neural Networks" by Bruch et al.].

    Args:
    labels: A `Tensor` of the same shape as `logits` representing graded
      relevance.
    logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
      ranking score of the corresponding item.
    weights: A scalar, a `Tensor` with shape [batch_size, 1] for list-wise
      weights, or a `Tensor` with shape [batch_size, list_size] for item-wise
      weights. If None, the weight of a list in the mini-batch is set to the sum
      of the labels of the items in that list.
    reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to
      reduce training loss over batch.
    name: A string used as the name for this loss.
    alpha: The exponent in the generalized sigmoid function.

    Returns:
    An op for the ApproxNDCG loss.
    """
    loss = losses_impl.ApproxNDCGLoss(name, params={'alpha': alpha})
    return loss.compute(labels, logits, weights, reduction)


def _approx_mrr_loss(labels,
                     logits,
                     weights=None,
                     reduction=losses_impl.Reduction.SUM,
                     name=None,
                     alpha=10.):
    """Computes ApproxMRR loss.

    ApproxMRR ["A general approximation framework for direct optimization of
    information retrieval measures" by Qin et al.] is a smooth approximation
    to MRR.

    Args:
    labels: A `Tensor` of the same shape as `logits` representing graded
      relevance.
    logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
      ranking score of the corresponding item.
    weights: A scalar, a `Tensor` with shape [batch_size, 1] for list-wise
      weights, or a `Tensor` with shape [batch_size, list_size] for item-wise
      weights. If None, the weight of a list in the mini-batch is set to the sum
      of the labels of the items in that list.
    reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to
      reduce training loss over batch.
    name: A string used as the name for this loss.
    alpha: The exponent in the generalized sigmoid function.

    Returns:
    An op for the ApproxMRR loss.
    """
    loss = losses_impl.ApproxMRRLoss(name, params={'alpha': alpha})
    return loss.compute(labels, logits, weights, reduction)
