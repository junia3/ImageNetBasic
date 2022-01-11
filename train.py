import os
import torch
import torch.optim as optim
import copy
import torch.nn.functional as F
import pickle as pkl

# logging information
def log_progress(epoch, num_epoch, iteration, num_data, batch_size, loss, acc):
    progress = int(iteration / (num_data // batch_size) * 100 // 4)
    print("Epoch : %d/%d >>>> train : %d/%d(%.2f%%) ( " % (epoch, num_epoch, iteration, num_data // batch_size, iteration / (num_data // batch_size) * 100)
          + '=' * progress + '>' + ' ' * (25 - progress) + " ) loss : %.6f, accuracy : %.2f%%" % (loss, acc * 100), end='\r')

# training model
def train_model(model, train_dataloader, val_dataloader, epochs, criterion, optimizer, scheduler,
                LOG_DIR=None, CHECKPOINT=None, device=None, model_name=None, crop=False):

    # set log directory
    if LOG_DIR is None:
        LOG_DIR = os.path.join(os.getcwd(), "logs")

    # set checkpoint directory
    if CHECKPOINT is None:
        CHECKPOINT = os.path.join(os.getcwd(), "checkpoints")

    # model name(defined on main)
    if model_name is None:
        model_name = input("Enter the model name : ")

    # set device if not specified
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_multiple = 1
    if crop:
       batch_multiple = 5
    #info for log
    num_train = len(train_dataloader.dataset)

    # to memorize the best model
    loss_history = {"train" : [], "val" : []}
    acc_history = {"train":[], "val" : []}
    best_acc = 0.0
    best_loss = float('inf')
    best_model_state = copy.deepcopy(model.state_dict())

    # model training
    model.to(device) # model on GPU/CPU
    for epoch in range(epochs):
        train_loss = 0.0
        train_acc = 0.0
        model.train()
        # train model with training data
        for idx, (image, label) in enumerate(train_dataloader):
            image, label = image.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(image)
            
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            _, prediction = torch.max(output, axis=1)
            accuracy = float(torch.sum(torch.eq(prediction, label)))/len(prediction)
            train_acc = (train_acc*idx + accuracy)/(idx+1)
            train_loss = (train_loss*idx + loss)/(idx+1) 
            log_progress(epoch, epochs, idx, num_train*batch_multiple, len(prediction), train_loss, train_acc)

        # validate model with validation data
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_acc = 0.0
            for idx, (image, label) in enumerate(val_dataloader):
                image, label = image.to(device), label.to(device)
                output = model(image)

                loss = criterion(output, label)
                _, prediction = torch.max(output, axis=1)
                accuracy = float(torch.sum(torch.eq(prediction, label)))/len(prediction)
                val_acc = (val_acc*idx+accuracy)/(idx+1)
                val_loss = (val_loss*idx+loss)/(idx+1)
            
            loss_history['val'].append(val_loss)
            acc_history['val'].append(val_acc)
            loss_history['train'].append(train_loss)
            acc_history['train'].append(train_acc)

            print("(Finish) Epoch : %d/%d >>>> avg_loss : %.6f,  avg_acc : %.2f%%   Validation loss : %.6f, Validation accuracy : %.2f%%"
                   %(epoch, epochs, train_loss, train_acc*100, val_loss, val_acc * 100))

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), os.path.join(CHECKPOINT, model_name+"_e{}.pkl".format(epoch+1)))
                print("Best model is copied..")
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

    with open(model_name+"_loss_history.pkl", 'wb') as f:
        pkl.dump(loss_history, f)
   
    with open(model_name+"_acc_history.pkl", 'wb') as f:
        pkl.dump(acc_history, f)
