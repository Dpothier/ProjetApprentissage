import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from deeplib.history import History
from deeplib.data import train_valid_loaders
from torchtext import data


def validate(model, val_loader, use_gpu=False):
    true_process = []
    true_material = []
    true_task = []

    pred_process = []
    pred_material = []
    pred_task = []

    val_loss_process = []
    val_loss_material = []
    val_loss_task = []

    criterion = nn.CrossEntropyLoss()
    model.eval()

    for batch in val_loader:

        texts = batch.texts
        process_targets = batch.process_tags
        material_targets = batch.material_tags
        task_targets = batch.task_tags

        if use_gpu:
            texts = texts.cuda()
            process_targets = process_targets.cuda()
            material_targets = material_targets.cuda()
            task_targets = task_targets.cuda()

        out_process, out_material, out_task = model(texts)

        process_predictions = out_process.max(dim=1)[1]
        material_predictions = out_material.max(dim=1)[1]
        task_predictions = out_task.max(dim=1)[1]

        process_loss = 0
        material_loss = 0
        task_loss = 0
        for i in range(out_process.shape[2]):
            process_preds = out_process[:, :, i]
            material_preds = out_material[:, :, i]
            task_preds = out_task[:, :, i]

            process_target = process_targets[i, :]
            material_target = material_targets[i, :]
            task_target = task_targets[i, :]

            process_loss += criterion(process_preds, process_target)[0]
            material_loss += criterion(material_preds, material_target)[0]
            task_loss += criterion(task_preds, task_target)[0]

        val_loss_process.append(process_loss)
        val_loss_material.append(material_loss)
        val_loss_task.append(task_loss)

        print(process_targets.shape)
        print(process_predictions.shape)
        true_process.extend(process_targets.data.cpu().numpy().tolist())
        pred_process.extend(process_predictions.data.cpu().numpy().tolist())

        true_material.extend(material_targets.data.cpu().numpy().tolist())
        pred_material.extend(material_predictions.data.cpu().numpy().tolist())

        true_task.extend(task_targets.data.cpu().numpy().tolist())
        pred_task.extend(task_predictions.data.cpu().numpy().tolist())

    return (accuracy_score(true_process, pred_process) * 100, sum(val_loss_process) / len(val_loss_process)),\
           (accuracy_score(true_material, pred_material) * 100, sum(val_loss_material) / len(val_loss_material)),\
           (accuracy_score(true_task, pred_task) * 100, sum(val_loss_task) / len(val_loss_task))


def train(model, dataset, n_epoch, batch_size, learning_rate, use_gpu=False):
    history_process = History()
    history_material = History()
    history_task = History()

    criterion = nn.NLLLoss2d()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    train, val = dataset
    train_iter = data.Iterator(
        train, batch_size=32, device=-1 if use_gpu is False else None, repeat=False)
    val_iter = data.Iterator(val, batch_size=32 , device=-1 if use_gpu is False else None, repeat=False)

    for i in range(n_epoch):
        model.train()
        for batch in train_iter:
            print("Starting batch")
            texts = batch.texts
            process_targets = batch.process_tags
            material_targets = batch.material_tags
            task_targets = batch.task_tags

            if use_gpu:
                texts = texts.cuda()
                process_targets = process_targets.cuda()
                material_targets = material_targets.cuda()
                task_targets = task_targets.cuda()

            optimizer.zero_grad()
            print("Before prediction")
            out_process,  out_material, out_task = model(texts)
            print("Before loss calculation")
            for i in range(out_process.shape[2]):
                process_preds = out_process[:, :, i]
                material_preds = out_material[:, :, i]
                task_preds = out_task[:, :, i]

                process_target = process_targets[i, :]
                material_target = material_targets[i, :]
                task_target = task_targets[i, :]

                process_loss = criterion(process_preds, process_target)
                material_loss = criterion(material_preds, material_target)
                task_loss = criterion(task_preds, task_target)

                process_loss.backward(retain_graph=True)
                material_loss.backward(retain_graph=True)
                task_loss.backward(retain_graph=True)
            print("Before optimization")
            optimizer.step()
            print("End batch")

        train_process, train_material, train_task = validate(model, train_iter, use_gpu)
        val_process, val_material, val_task = validate(model, val_iter, use_gpu)
        history_process.save(train_process[0], val_process[0], train_process[1], val_process[1])
        history_material.save(train_material[0], val_material[0], train_material[1], val_material[1])
        history_task.save(train_task[0], val_task[0], train_task[1], val_task[1])
        print('Epoch {} process - Train acc: {:.2f} - Val acc: {:.2f} - Train loss: {:.4f} - Val loss: {:.4f}'
              .format(i, train_process[0], val_process[0], train_process[1], val_process[1]))
        print('Epoch {} material - Train acc: {:.2f} - Val acc: {:.2f} - Train loss: {:.4f} - Val loss: {:.4f}'
              .format(i, train_material[0], val_material[0], train_material[1], val_material[1]))
        print('Epoch {} task - Train acc: {:.2f} - Val acc: {:.2f} - Train loss: {:.4f} - Val loss: {:.4f}'
              .format(i, train_task[0], val_task[0], train_task[1], val_task[1]))
    return history_process, history_material, history_task
