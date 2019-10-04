import torch.nn as nn

class PretrainLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.criterion = nn.NLLLoss(ignore_index=0)


    def __call__(self, preds, true):
        sentence_pred, words_pred = preds
        sentence_true, words_true = true

        next_loss = self.criterion(sentence_pred, sentence_true)
        mask_loss = self.criterion(words_pred.transpose(1, 2), words_true)
        loss = next_loss + mask_loss

        return loss
