import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder

from tqdm.auto import tqdm

def dataset():

    train_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(45),
        transforms.ToTensor(),
    ])

    test_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    batch_size = 128

    train_set = DatasetFolder("OxFlower17/train", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
    test_set = DatasetFolder("OxFlower17/test", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

class TransferNet(nn.Module):
    def __init__(self, model, input_dim=1000, output_dim=17):
        super(TransferNet, self).__init__()

        self.pre_layers = nn.Sequential(model)
        # for param in self.parameters():
        #     param.requires_grad = False
        self.last_layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.last_layer(x)
        return x

def train(new_model,train_loader,n_epochs):

    for epoch in range(n_epochs):

        new_model.train()

        train_loss = []
        train_accs = []

        for batch in tqdm(train_loader):
            imgs, labels = batch

            logits = new_model(imgs.to(device))

            loss = criterion(logits, labels.to(device))

            optimizer.zero_grad()

            loss.backward()

            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            optimizer.step()

            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            train_loss.append(loss.item())
            train_accs.append(acc)

        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)

        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

def text(new_model,test_loader):

    new_model.eval()

    test_loss = []
    test_accs = []

    for batch in tqdm(test_loader):
        imgs, labels = batch

        with torch.no_grad():
            logits = new_model(imgs.to(device))

        loss = criterion(logits, labels.to(device))

        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        test_loss.append(loss.item())
        test_accs.append(acc)

    test_loss = sum(test_loss) / len(test_loss)
    test_acc = sum(test_accs) / len(test_accs)

    print(f"test | loss = {test_loss:.5f}, acc = {test_acc:.5f}")

if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = timm.create_model('vit_base_patch16_224', pretrained=True)

    new_model = TransferNet(model)
    # print(new_model)

    new_model = new_model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, new_model.parameters()), lr=0.0003, weight_decay=1e-5)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)

    train_loader, test_loader = dataset()

    n_epochs = 10

    train(new_model,train_loader,n_epochs)

    text(new_model, test_loader)