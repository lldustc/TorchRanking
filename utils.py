import torch

def logical_and(tensor1, tensor2):
    return torch.add(tensor1.long(), tensor2.long()) > 1


def logical_or(tensor1, tensor2):
    return torch.add(tensor1.long(), tensor2.long()) > 0


def logical_xor(tensor1, tensor2):
    return torch.add(tensor1.long(), tensor2.long()) == 1

def is_label_valid(labels):
    """Returns a boolean `Tensor` for label validity."""
    return labels >= 0

def sort_by_scores(scores, features_list, topn=None,shuffle_ties=True):
    scores = scores.float()
    list_size = scores.shape[1]
    if topn is None:
        topn = list_size
    topn = torch.min(torch.tensor([list_size, topn])).item()
    shuffle_ind=None
    if shuffle_ties:
        shuffle_ind=torch.argsort(torch.rand(scores.shape))
        scores=torch.gather(scores,1,shuffle_ind)
    _,indices=torch.topk(scores,topn,sorted=True)
    if shuffle_ind is not None:
        indices=torch.gather(shuffle_ind,1,indices)
    return [torch.gather(f, 1, indices) for f in features_list]


def inverse_max_dcg(labels,
                    gain_fn=lambda labels: torch.pow(2.0, labels) - 1.,
                    rank_discount_fn=lambda rank: 1. / torch.log1p(rank),
                    topn=None):
    ideal_sorted_labels, = sort_by_scores(labels, [labels], topn=topn)
    rank = torch.arange(1, ideal_sorted_labels.shape[1] + 1, dtype=torch.float)
    discounted_gain = gain_fn(ideal_sorted_labels.float()) * rank_discount_fn(rank)
    discounted_gain = discounted_gain.sum(dim=1, keepdim=True)
    return torch.where(discounted_gain > 0., 1. / discounted_gain,
                       torch.zeros_like(discounted_gain))


def sorted_ranks(scores,shuffle_ties=True):
    """Returns an int `Tensor` as the ranks (1-based) after sorting scores.

    Example: Given scores = [[1.0, 3.5, 2.1]], the returned ranks will be [[3, 1,
    2]]. It means that scores 1.0 will be ranked at position 3, 3.5 will be ranked
    at position 1, and 2.1 will be ranked at position 2.

    Args:
    scores: A `Tensor` of shape [batch_size, list_size] representing the
      per-example scores.
    shuffle_ties: See `sort_by_scores`.

    Returns:
    A 1-based int `Tensor`s as the ranks.
    """
    batch_size, list_size = scores.shape
    # The current position in the list for each score.
    positions = torch.unsqueeze(torch.arange(list_size), 0).repeat(batch_size, 1)
    # For score [[1.0, 3.5, 2.1]], sorted_positions are [[1, 2, 0]], meaning the
    # largest score is at poistion 1, the second is at postion 2 and third is at
    # position 0.
    sorted_positions = sort_by_scores(scores, [positions],shuffle_ties=shuffle_ties)[0]
    # The indices of sorting sorted_postions will be [[2, 0, 1]] and ranks are
    # 1-based and thus are [[3, 1, 2]].
    ranks = torch.argsort(sorted_positions) + 1
    return ranks


def shuffle_valid_indices(is_valid):
    """Returns a shuffle of indices with valid ones on top."""
    return organize_valid_indices(is_valid, shuffle=True)


def organize_valid_indices(is_valid, shuffle=True):
    """Organizes indices in such a way that valid items appear first.

    Args:
    is_valid: A boolen `Tensor` for entry validity with shape [batch_size,
      list_size].
    shuffle: A boolean indicating whether valid items should be shuffled.
    seed: An int for random seed at the op level. It works together with the
      seed at global graph level together to determine the random number
      generation. See `tf.set_random_seed`.

    Returns:
    A tensor of indices with shape [batch_size, list_size, 2]. The returned
    tensor can be used with `tf.gather_nd` and `tf.scatter_nd` to compose a new
    [batch_size, list_size] tensor. The values in the last dimension are the
    indices for an element in the input tensor.
    """
    assert is_valid.dim() >= 2
    output_shape = tuple(is_valid.shape)

    if shuffle:
        values = torch.rand(is_valid.shape)
    else:
        values = torch.ones_like(is_valid).float() * torch.arange(is_valid.shape[1] - 1, -1, -1, dtype=torch.float)

    rand = torch.where(is_valid, values, torch.ones_like(is_valid).float() * -1e-6)
    # shape(indices) = [batch_size, list_size]
    indices = torch.argsort(rand, descending=True)
    return indices


def approx_ranks(logits, alpha=10.):
    r"""Computes approximate ranks given a list of logits.

    Given a list of logits, the rank of an item in the list is simply
    one plus the total number of items with a larger logit. In other words,

    rank_i = 1 + \sum_{j \neq i} I_{s_j > s_i},

    where "I" is the indicator function. The indicator function can be
    approximated by a generalized sigmoid:

    I_{s_j < s_i} \approx 1/(1 + exp(-\alpha * (s_j - s_i))).

    This function approximates the rank of an item using this sigmoid
    approximation to the indicator function. This technique is at the core
    of "A general approximation framework for direct optimization of
    information retrieval measures" by Qin et al.

    Args:
    logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
      ranking score of the corresponding item.
    alpha: Exponent of the generalized sigmoid function.

    Returns:
    A `Tensor` of ranks with the same shape as logits.
    """
    list_size = logits.shape[1]
    x = torch.unsqueeze(logits, 2)
    y = torch.unsqueeze(logits, 1)
    pairs = torch.sigmoid(alpha * (y - x))
    return torch.sum(pairs, dim=-1) + .5


def reshape_to_2d(tensor):
    rank = tensor.dim()
    if rank != 2:
        if rank >= 3:
            tensor = tensor.view(tensor.shape[0:2])
        else:
            while tensor.dim() < 2:
                tensor = torch.unsqueeze(tensor, -1)
    return tensor
