import torch
import torch.nn
import time


# set flags /seeds

class Options:
    path_to_checkpoint = "."
    resume = False
    epoches = 1
    eval_every_epoches = 1


class ModeKeys:
    EVAL = "EVAL"
    PREDICT = "PREDICT"
    TRAIN = "TRAIN"

class Head:
    def __init__(self, loss_fn=None, eval_metric_fns=None):
        self.loss_fn = loss_fn
        self._eval_metric_fns = eval_metric_fns or {}

    def run(self, mode, labels=None, logits=None, features=None):
        if mode == ModeKeys.EVAL:
            loss = self.loss_fn(labels, logits, features)
            if len(self._eval_metric_fns) == 0:
                raise ValueError("no metrics function")
            metrics_values = {name: fn(labels, logits, features)
                              for name, fn in self._eval_metric_fns.items()}
            return loss, metrics_values

        if mode == ModeKeys.TRAIN:
            return self.loss_fn(labels, logits, features)

        if mode == ModeKeys.PREDICT:
            return logits


def train(options, model, optim, head,
          train_dataloader, test_dataloader, features):
    def load_checkpoint(self, path):
        return torch.load(path)

    def save_checkpoint(self, model, epoch, n_iter, optim, path):
        ckpt = {}
        ckpt["model"] = model.state_dict()
        ckpt["epoch"] = epoch
        ckpt["n_iter"] = n_iter
        ckpt["optim"] = optim.state_dict()
        torch.save(ckpt, path)

    def train_resume():
        ckpt = load_checkpoint(options.path_to_checkpoint)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", 0)
        start_n_iter = ckpt.get("n_iter", 0)
        optim.load_state_dict(ckpt["optim"])
        print("checkpoint restored.")
        return start_epoch, start_n_iter

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    start_n_iter = 0
    start_epoch = 0
    if options.resume:
        start_epoch, start_n_iter = train_resume()
    n_iter = start_n_iter
    for epoch in range(start_epoch, options.epoches):
        model.train()
        start_time = time.time()
        for i, (x, labels) in enumerate(train_dataloader):
            if use_cuda:
                x = x.cuda()
                labels = labels.cuda()
            prepare_time = time.time() - start_time
            optim.zero_grad()
            loss, _ = head.run(mode=ModeKeys.TRAIN,
                               labels=labels,
                               logits=model(x),
                               features=features)
            loss.backward()
            optim.step()
            process_time = time.time() - start_time - prepare_time
            start_time = time.time()
        if epoch % options.eval_every_epoches == options.eval_every_epoches - 1:
            eval(model, head, test_dataloader, features)
            save_checkpoint(model, epoch, n_iter, optim, options.path_to_checkpoint)


def eval(model, head, test_dataloader, features):
    model.eval()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    for i, (x, labels) in enumerate(test_dataloader):
        if use_cuda:
            x = x.cuda()
            labels = labels.cuda()
        loss, metric_values = head.run(mode=ModeKeys.EVAL,
                                       labels=labels,
                                       logits=model(x),
                                       features=features)
