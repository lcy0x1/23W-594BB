import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

from data_processing.dataloader import LoaderCreator, DataParam, AudioMNIST
from data_processing.dataloader_test_model import CNN2DAudioClassifier

DATAPATH = './Datasets/audio/'

if torch.cuda.is_available():
    print("Using CUDA device")
    device = torch.device("cuda:0")
else:
    print("Using CPU")
    device = torch.device('cpu')

train_param = DataParam(0.8, 64, shuffle=True)
val_param = DataParam(0.12, 64, shuffle=False)
test_param = DataParam(0.08, 32, shuffle=False)

train_dl, val_dl, test_dl = LoaderCreator(DATAPATH).create_loaders(
    train_param,
    val_param,
    test_param)

model = CNN2DAudioClassifier()
N_EPOCHS = 2
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                max_lr=0.001,
                                                steps_per_epoch=int(len(train_dl)),
                                                epochs=N_EPOCHS,
                                                anneal_strategy='linear')

def evaluate(model, val_dl):
    running_loss = 0.0
    correct_prediction = 0
    total_prediction = 0

    for data in tqdm(val_dl):
        inputs, labels = data[0].to(device), data[1].to(device)

        inputs_m, inputs_s = inputs.mean(), inputs.std()
        inputs = (inputs - inputs_m) / inputs_s

        optimizer.zero_grad()

        outputs = model(inputs.to(device))
        loss = criterion(outputs, labels.to(device))

        running_loss += loss.item()

        _, prediction = torch.max(outputs, 1)
        correct_prediction += (prediction == labels).sum().item()
        total_prediction += prediction.shape[0]

    num_batches = len(val_dl)
    avg_loss = running_loss / num_batches
    acc = correct_prediction / total_prediction

    return acc, avg_loss


def training(model, train_dl, val_dl, num_epochs,
             criterion, optimizer, scheduler):
    losses = []
    val_losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0

        for data in tqdm(train_dl):
            inputs, labels = data[0].to(device), data[1].to(device)

            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            optimizer.zero_grad()

            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            _, prediction = torch.max(outputs, 1)
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        acc = correct_prediction / total_prediction

        v_acc, v_loss = evaluate(model.to(device), val_dl)

        print("Epoch: %d, Loss: %.4f, Train Accuracy: %.2f, Val. Loss: %.4f, Val. Accuracy: %.2f" % (
            epoch + 1, avg_loss, acc, v_loss, v_acc
        ))

        losses.append(avg_loss)
        val_losses.append(v_loss)

    return losses, val_losses

def dataloader_test():
    #losses, val_losses = training(model, train_dl, val_dl, N_EPOCHS, criterion, optimizer, scheduler)
    #evaluate(model, val_dl)
    #plt.plot(range(len(losses)), losses, range(len(val_losses)), val_losses)
    #plt.legend(['Training Loss', 'Validation Loss'])
    #plt.show()
    dataset = AudioMNIST(datapath=DATAPATH)
    dataset[0].shape

if __name__ == "__main__":
    dataloader_test()
