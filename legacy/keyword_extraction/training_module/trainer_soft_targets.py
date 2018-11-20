from training_module.trainer import Trainer
import torch.nn as nn
import torch
from torch.autograd import Variable
from helper.softmax import softmax

class TrainerSoftTarget(Trainer):

    def __init__(self, number_of_classes, max_target, use_gpu=False):
        self.number_of_classes = number_of_classes
        self.max_target = max_target
        self.use_gpu = use_gpu



    def _calculate_loss(self, out_process, process_targets, out_material, material_targets, out_task, task_targets, class_weight=None, use_gpu=False):
        criterion = nn.BCEWithLogitsLoss(weight=class_weight)
        if use_gpu:
            criterion = criterion.cuda()

        process_loss = criterion(out_process.permute(0, 2, 1), process_targets)
        material_loss = criterion(out_material.permute(0, 2, 1), material_targets)
        task_loss = criterion(out_task.permute(0, 2, 1), task_targets)

        return process_loss, material_loss, task_loss


    def _prepare_targets(self, batch):
        process_targets = self._soft_one_hot_encode(batch.process_tags.permute(1, 0), self.max_target, self.number_of_classes, self.use_gpu)
        material_targets = self._soft_one_hot_encode(batch.material_tags.permute(1, 0), self.max_target, self.number_of_classes, self.use_gpu)
        task_targets = self._soft_one_hot_encode(batch.task_tags.permute(1, 0), self.max_target, self.number_of_classes, self.use_gpu)

        return process_targets, material_targets, task_targets

    def _soft_one_hot_encode(self, class_valued_tensor, max_value, number_of_classes, use_gpu):
        class_valued_tensor = class_valued_tensor.cpu()

        min_value = (1 - max_value) / (number_of_classes - 1)
        batch_size = class_valued_tensor.shape[0]
        sequence_size = class_valued_tensor.shape[1]

        one_hot_encode = torch.ones((batch_size, sequence_size, number_of_classes)) * min_value
        one_hot_encode.scatter_(2, class_valued_tensor.contiguous().data.view(batch_size, sequence_size, 1), max_value)
        one_hot_encode = Variable(one_hot_encode)
        if use_gpu:
            one_hot_encode = one_hot_encode.cuda()
        return one_hot_encode

    def _compute_confidence(self, process_targets, out_process, probabilities):
        hard_target = process_targets.max(dim=2)[1]
        super()._compute_confidence(hard_target, out_process, probabilities)


    def _compute_accuracy(self, process_targets, process_predictions, material_targets, material_predictions, task_targets, task_predictions, true, pred):
        hard_process_target = process_targets.max(dim=2)[1]
        hard_material_target = material_targets.max(dim=2)[1]
        hard_task_target = task_targets.max(dim=2)[1]

        super()._compute_accuracy(hard_process_target, process_predictions,
                                  hard_material_target, material_predictions,
                                  hard_task_target, task_predictions,
                                  true, pred)