from torch.nn.modules.loss import _WeightedLoss
import torch.nn as nn
import torch


class SoftCrossEntropyLoss(nn.Module):
    r"""This criterion combines :func:`nn.LogSoftmax` and :func:`nn.NLLLoss` in one single class while allowing for soft target encoding.

    It is useful when training a classification problem with `C` classes.
    The soft target encoding may help in preventing overfitting.
    If provided, the optional argument :attr:`weight` should be a 1D `Tensor`
    assigning weight to each of the classes.
    This is particularly useful when you have an unbalanced training set.

    The `input` is expected to contain scores for each class.

    `input` has to be a Tensor of size either :math:`(minibatch, C)` or
    :math:`(minibatch, C, d_1, d_2, ..., d_K)`
    with :math:`K \geq 2` for the `K`-dimensional case (described later).

    This criterion expects a class index (0 to `C-1`) as the
    `target` for each value of a 1D tensor of size `minibatch`

    The loss can be described as:

    .. math::
        \text{loss}(x, class) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
                       = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    or in the case of the `weight` argument being specified:

    .. math::
        \text{loss}(x, class) = weight[class] \left(-x[class] + \log\left(\sum_j \exp(x[j])\right)\right)

    The losses are averaged across observations for each minibatch.

    Can also be used for higher dimension inputs, such as 2D images, by providing
    an input of size :math:`(minibatch, C, d_1, d_2, ..., d_K)` with :math:`K \geq 2`,
    where :math:`K` is the number of dimensions, and a target of appropriate shape
    (see below).


    Args:
        weight (Tensor, optional): a manual rescaling weight given to each class.
            If given, has to be a Tensor of size `C`
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When `size_average` is
            ``True``, the loss is averaged over non-ignored targets.
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: 'mean'

    Shape:
        - Input: :math:`(N, C)` where `C = number of classes`, or
            :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 2`
            in the case of `K`-dimensional loss.
        - Target: :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`, or
            :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 2` in the case of
            K-dimensional loss.
        - Output: scalar. If reduce is ``False``, then the same size
            as the target: :math:`(N)`, or
            :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 2` in the case
            of K-dimensional loss.

    Examples::

        >>> loss = nn.SoftCrossEntropyLoss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(5)
        >>> output = loss(input, target)
        >>> output.backward()
    """
    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self,soft_target_fct, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(SoftCrossEntropyLoss, self).__init__()
        self.soft_target_fct = soft_target_fct
        self.register_buffer("weight", weight)
        self.size_average = size_average
        self.reduce = reduce
        self.reduction = reduction
        self.ignore_index = ignore_index

        self.eval_loss = nn.CrossEntropyLoss(weight=weight, size_average=size_average, ignore_index=ignore_index,
                                             reduce=reduce, reduction=reduction)


    def forward(self, input, target):
        if self.training:
            soft_target = self.soft_target_fct(target)

            logsoftmax = nn.LogSoftmax()
            if self.size_average:
                loss = torch.mean(torch.sum(-soft_target * logsoftmax(input) * self.weight, dim=1))
                return loss
            else:
                loss = -soft_target * logsoftmax(input) * self.weight
                loss = torch.sum(torch.sum(loss , dim=1))
                return loss
        else:
            return self.eval_loss(input, target)

class SoftenTargets():
    def __init__(self, number_of_classes, upper_bound):
        self.number_of_classes = number_of_classes
        self.upper_bound = upper_bound

    def __call__(self, hard):
        number_of_exemples = hard.shape[0]
        off_target_value = (1 - self.upper_bound) / (self.number_of_classes - 1)
        soft = torch.Tensor([off_target_value]).repeat(number_of_exemples, self.number_of_classes)
        j = torch.arange(hard.size(0))
        soft[j, hard] = self.upper_bound

        if hard.is_cuda:
            soft = soft.cuda()

        return soft

