import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.autograd import Variable
from Unet_model import segment_data, UNet
import matplotlib.pyplot as plt


def dice_similarity_coefficient(y_true, y_predicted, axis=(2,3), epsilon=0.00001):
    dice_numerator = 2. * torch.sum(y_true * y_predicted, dim=axis) + epsilon
    dice_denominator = torch.sum(y_true, dim=axis) + torch.sum(y_predicted, dim=axis) + epsilon
    dice_coefficient = torch.mean((dice_numerator)/(dice_denominator))
    return dice_coefficient


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = np.inf

    def early_stop(self, loss):
        if loss <=self.min_loss:
            self.min_loss = loss
            self.counter = 0
        elif loss > (self.min_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


if __name__=="__main__":
    batch_size = 8
    num_epochs = 100

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet()
    model.to(device)

    image_path = 'image_data'
    mask_path = 'mask_data'
    model_path = 'pretrained_weights/unet_pretrained.pth'

    seg_data = segment_data(image_path, mask_path)
    val_size = int(0.1 * len(seg_data))
    train_size = int(0.8 * len(seg_data))
    test_size = len(seg_data) - (train_size + val_size)
    train_dataset, val_dataset, test_dataset = random_split(seg_data, [train_size, val_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # dataset = DataLoader(segment_data(image_path, mask_path),
    #                      batch_size=batch_size,
    #                      shuffle=True)
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}
    phase = 'train'
    # for dat in train_dataloader:
    #     image = dat['image']
    #     mask = dat['mask']
    #
    #     print(image.shape)
    #     print()

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    earlystopper = EarlyStopper(5)

    train_metrics = {'loss': [], 'dsc': []}
    val_metrics = {'loss': [], 'dsc': []}
    best_loss = np.inf
    tr_loss, vl_loss, tr_dsc, vl_dsc = [], [], [], []

    print("Training........................")

    for epoch in range(0, num_epochs):

        for phase in ['train', 'val']:
            running_loss = 0.0
            dsc_avg = 0.0

            if phase == 'train':
                model.train()
            else:
                model.eval()

            for dat in dataloaders[phase]:
                image = Variable(dat['image'].type(Tensor))
                mask = Variable(dat['mask'].type(Tensor))

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    predicted = model(image)
                    loss = criterion(predicted, mask)
                    dsc = dice_similarity_coefficient(mask, predicted)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                dsc_avg += dsc.item()
                running_loss += loss.item()
            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_dsc = dsc_avg / len(dataloaders[phase])
            if phase == 'train':
                train_metrics['loss'] = epoch_loss
                train_metrics['dsc'] = epoch_dsc
            elif phase == 'val':
                val_metrics['loss'] = epoch_loss
                val_metrics['dsc'] = epoch_dsc

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), model_path)

        train_l, train_dsc = train_metrics['loss'], train_metrics['dsc']
        val_l, val_dsc = val_metrics['loss'], val_metrics['dsc']
        tr_loss.append(train_l)
        vl_loss.append(val_l)
        tr_dsc.append(train_dsc)
        vl_dsc.append(val_dsc)

        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'training loss: {train_l}, training dsc: {train_dsc}')
        print(f'validation loss: {val_l}, validation dsc: {val_dsc}')

        if earlystopper.early_stop(val_l):
            break

    font1 = {'family': 'serif', 'color': 'blue', 'size': 20}
    font2 = {'family': 'serif', 'color': 'darkred', 'size': 15}
    plt.figure(1)
    plt.plot(tr_loss, label='training loss')
    plt.plot(vl_loss, label='validation loss')
    plt.title('Loss Plot', fontdict=font1)
    plt.xlabel('Epoch', fontdict=font2)
    plt.ylabel('Loss', fontdict=font2)
    plt.legend()
    plt.savefig('graphs/loss_graph.png')

    plt.figure(2)
    plt.plot(tr_dsc, label='training accuracy')
    plt.plot(vl_dsc, label='validation accuracy')
    plt.title('Accuracy Plot', fontdict=font1)
    plt.xlabel('Epoch', fontdict=font2)
    plt.ylabel('DSC', fontdict=font2)
    plt.legend()
    plt.savefig('graphs/DSC_plot.png')

    test_model = UNet()
    test_model.load_state_dict(torch.load(model_path))
    test_model.to(device)
    test_model.eval()

    test_dsc = 0.0

    with torch.no_grad():
        for dat in test_dataset:
            image = Variable(dat['image'].type(Tensor))
            mask = Variable(dat['mask'].type(Tensor))

            predicted = test_model(image)

            dsc = dice_similarity_coefficient(mask, predicted)

            test_dsc += dsc.item()

        total_dsc = test_dsc / len(test_dataset)

    print(f'Test DSC accuracy is: {total_dsc}')






