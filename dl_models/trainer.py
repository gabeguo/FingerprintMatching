# Thanks https://github.com/adambielski/siamese-triplet`

import torch
import numpy as np
import torch.nn as nn


def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[],
        start_epoch=0, early_stopping_interval=10, temp_model_path='temp.pth'):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss

    Returns: epoch stopped on and final val loss
    Loads the best validation weights into the model
    """

    torch.save(model.state_dict(), temp_model_path)
    best_val_loss = 100
    best_val_epoch = -1
    
    if scheduler is not None:
        for epoch in range(0, start_epoch):
            scheduler.step()

    past_val_losses = []
    past_train_losses = []

    for epoch in range(start_epoch, n_epochs):
        print('current epoch:', epoch)

        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics)
        past_train_losses.append(train_loss)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics)
        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch
            torch.save(model.state_dict(), temp_model_path)
        past_val_losses.append(val_loss)
        if len(past_val_losses) > early_stopping_interval and val_loss > sum(past_val_losses[-early_stopping_interval:]) / len(past_val_losses[-early_stopping_interval:]):
            print('val loss no longer decreasing - stop training')

            # load best weights
            model.load_state_dict(torch.load(temp_model_path))
            model.eval()
            
            from datetime import datetime
            datetime_str = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
            with open('/data/therealgabeguo/results/train_res_{}.txt'.format(datetime_str), 'w') as fout:
                fout.write('epoch: ' + str([epoch for epoch in range(start_epoch, epoch + 1)]) + '\n')
                fout.write('train loss: ' + str(past_train_losses) + '\n')
                fout.write('val loss: ' + str(past_val_losses) + '\n')
                fout.write('stopped on epoch {}\n'.format(epoch))

            return best_val_epoch, best_val_loss

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        print(message)
        
        if scheduler is not None:
            scheduler.step()

    from datetime import datetime
    datetime_str = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    with open('/data/therealgabeguo/results/train_res_{}.txt'.format(datetime_str), 'w') as fout:
        fout.write('epoch: ' + str([epoch for epoch in range(start_epoch, n_epochs)]) + '\n')
        fout.write('train loss: ' + str(past_train_losses) + '\n')
        fout.write('val loss: ' + str(past_val_losses) + '\n')

    # load best weights
    model.load_state_dict(torch.load(temp_model_path))
    model.eval()
    return best_val_epoch, best_val_loss


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics):
    for metric in metrics:
        metric.reset()

    if cuda:
        model = model.to(cuda)
    model.train()
    losses = []
    total_loss = 0
    n_nonzero_losses = 0

    # individual losses
    individual_loss_fn = nn.TripletMarginLoss(margin=loss_fn.margin, reduction='none')
    # individual losses

    for batch_idx, (data, target, filepaths) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda(device=cuda) for d in data)
            if target is not None:
                target = torch.tensor([int(item) for item in target]).cuda(device=cuda)


        optimizer.zero_grad()
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        #print(loss_inputs)

        loss_inputs = list(loss_inputs)
        for i in range(len(loss_inputs)):
            loss_inputs[i] = loss_inputs[i].reshape(loss_inputs[i].size(dim=0), loss_inputs[i].size(dim=1))

        # individual losses
        individual_loss_outputs = individual_loss_fn(*loss_inputs)
        #print(loss_inputs[0].size())
        #print(individual_loss_outputs.size())
        #print('percentage with 0 loss:', 1 - (torch.count_nonzero(individual_loss_outputs) / loss_inputs[0].size(dim=0)).item())
        n_nonzero_losses += torch.count_nonzero(individual_loss_outputs)
        # individual losses

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())
            message += '\n\tPercent of losses that are zero: {}'.format(1 - (n_nonzero_losses / (log_interval * loss_inputs[0].size(dim=0))))
            print(message)
            losses = []
            n_nonzero_losses = 0

    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, cuda, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (data, target, filepaths) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda(device=cuda) for d in data)
                if target is not None:
                    #target = target.cuda()
                    target = torch.tensor([int(item) for item in target]).cuda(device=cuda)

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target
            
            # make loss function input be 2D instead of default 4D from resnet
            loss_inputs = list(loss_inputs)
            for i in range(len(loss_inputs)):
                loss_inputs[i] = loss_inputs[i].reshape(loss_inputs[i].size(dim=0), loss_inputs[i].size(dim=1))

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

    return val_loss, metrics
