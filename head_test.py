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

"""Tests for ranking head."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import unittest

import head as ranking_head
import metrics as metrics_lib


def _make_loss_fn(weights_feature_name=None):
  """Make a fake loss function."""

  def _loss_fn(labels, logits, features):
    """A fake loss function."""
    labels = labels.float()
    weights = features[
        weights_feature_name] if weights_feature_name is not None else torch.tensor(1.)
    loss = torch.sum(logits -labels) * torch.sum(weights)
    return loss

  return _loss_fn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer=nn.Linear(3,1)

    def forward(self, x):
        return self.layer(x)


class RankingHeadTest(unittest.TestCase):

  def setUp(self):
    self._default_logits = torch.tensor([[1., 3., 2.], [1., 2., 3.]])
    self._default_labels = torch.tensor([[0., 0., 1.], [0., 0., 2.]])
    self._default_loss = 9.
    self._default_weights = torch.tensor([1.] * 3)
    self._default_weights_feature_name = 'weights'
    self._default_weighted_loss = 27

  def test_eval(self):
      metric_fns = {
          'metric/precision@1':
              metrics_lib.make_ranking_metric_fn(
                  metrics_lib.RankingMetricKey.PRECISION, topn=1),
      }
      head=ranking_head.Head(loss_fn=_make_loss_fn(),eval_metric_fns=metric_fns)
      loss,metrics_values=head.run(ranking_head.ModeKeys.EVAL,self._default_labels,self._default_logits,features={})
      self.assertAlmostEqual(loss.item(),self._default_loss,5)

  def test_train(self):
      options = ranking_head.Options()
      head = ranking_head.Head(loss_fn=_make_loss_fn())
      loss=head.run(mode=ranking_head.ModeKeys.TRAIN,
               labels=self._default_labels,
               logits=self._default_logits,
               features={})

      self.assertAlmostEqual(loss.item(),self._default_loss)


if __name__=="__main__":
    unittest.main()


