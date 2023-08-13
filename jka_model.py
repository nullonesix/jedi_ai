from torchvision.io import read_image
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np
import argparse
# from skimage import io

img = read_image("screenshot.png")

# Step 1: Initialize model with the best available weights
weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
# model = mobilenet_v3_small(weights=weights)
# model.eval()

class Net(nn.Module):
    def __init__(self, in_channels=4, num_actions=18):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc4 = nn.Linear(36864, 512)
        # self.fc5 = nn.Linear(512, num_actions)
        self.w = nn.Linear(512, 2)
        self.a = nn.Linear(512, 2)
        self.s = nn.Linear(512, 2)
        self.d = nn.Linear(512, 2)
        self.f = nn.Linear(512, 2)
        self.e = nn.Linear(512, 2)
        self.r = nn.Linear(512, 2)
        self.space = nn.Linear(512, 2)
        self.ctrl = nn.Linear(512, 2)
        self.mouse_left = nn.Linear(512, 2)
        self.mouse_middle = nn.Linear(512, 2)
        self.mouse_right = nn.Linear(512, 2)
        self.mouse_deltaX = nn.Linear(512, 1)
        self.mouse_deltaY = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        w = F.log_softmax(self.w(x), dim=1)
        a = F.log_softmax(self.a(x), dim=1)
        s = F.log_softmax(self.s(x), dim=1)
        d = F.log_softmax(self.d(x), dim=1)
        f = F.log_softmax(self.f(x), dim=1)
        e = F.log_softmax(self.e(x), dim=1)
        r = F.log_softmax(self.r(x), dim=1)
        space = F.log_softmax(self.space(x), dim=1)
        ctrl = F.log_softmax(self.ctrl(x), dim=1)
        mouse_left = F.log_softmax(self.mouse_left(x), dim=1)
        mouse_middle = F.log_softmax(self.mouse_middle(x), dim=1)
        mouse_right = F.log_softmax(self.mouse_right(x), dim=1)
        mouse_deltaX = self.mouse_deltaX(x)
        mouse_deltaY = self.mouse_deltaY(x)
        return w, a, s, d, f, e, r, space, ctrl, mouse_left, mouse_middle, mouse_right, mouse_deltaX, mouse_deltaY
        # return self.fc5(x)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # data, target = data.to(device), target.to(device)
        data = data.to(device)
        # print(target.shape)
        # target = target.tolist()
        optimizer.zero_grad()
        output = model(data)
        # print(output.shape, target.shape)
        # loss = F.nll_loss(output, target)
        # print('output', output)
        # print('target', target)
        loss = 0.0
        pairs = list(zip(output, target))
        for pair in pairs[:-2]:
            # print('keys')
            # print(pair[0].type())
            # print(pair[1].type())
            # print('output shape', pair[0].shape)
            # print('target shape', pair[1].shape)
            loss += F.nll_loss(pair[0], torch.argmax(pair[1].to(device), dim=1)) #.to(device).type(torch.LongTensor))
        for pair in pairs[-2:]:
            # print('deltas')
            # print(pair[0])
            # print(pair[1])
            # print(pair[0].type())
            # print(pair[1].type())
            # print(pair[0].shape)
            # print(pair[1].shape)
            loss += F.mse_loss(pair[0], pair[1].type(torch.FloatTensor).to(device)) #.to(device).type(torch.FloatTensor))
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

# def test(model, device, test_loader):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
#             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#             correct += pred.eq(target.view_as(pred)).sum().item()

#     test_loss /= len(test_loader.dataset)

#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))

class JKADataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.inputs_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.inputs_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.inputs_frame.iloc[idx, 0])
        image = read_image(img_name)
        inputs = self.inputs_frame.iloc[idx, 1:]
        # inputs = np.array([inputs], dtype=float) #.reshape(-1, 2)
        inputs = np.array(inputs, dtype=float).tolist()
        inputs_transformed = []
        for x in inputs[:-2]:
            if x == 0:
                inputs_transformed.append(np.array([1, 0])) # OFF, on
            if x == 1:
                inputs_transformed.append(np.array([0, 1])) # off, ON
        inputs_transformed.append(np.array([inputs[-2]]))
        inputs_transformed.append(np.array([inputs[-1]]))
        inputs = inputs_transformed
        # inputs = (1, 2)
        # sample = {'image': image, 'inputs': inputs}

        if self.transform:
            # sample = self.transform(sample)
            image = self.transform(image)

        image = weights.transforms()(image) #.unsqueeze(0)

        data = image
        target = inputs
        # return sample
        return (data, target)


def main():

    parser = argparse.ArgumentParser(description='Jedi AI')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')
    args = parser.parse_args()

    device = torch.device('cpu')
    
    train_kwargs = {'batch_size': args.batch_size}

    transform=transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    dataset1 = JKADataset(csv_file='./data/data.csv', root_dir='./data/train/', transform=None)

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)

    model = Net(in_channels=3).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    
    args = parser.parse_args()
    
    epochs = 100
    for epoch in range(epochs):
        train(args, model, device, train_loader, optimizer, epoch)
    torch.save(model.state_dict(), 'jka_model3.pt')

if __name__ == '__main__':
    main()







# model = Net(in_channels=3)
# model.eval()

# # Step 2: Initialize the inference transforms
# preprocess = weights.transforms()

# t0 = time.time()
# # Step 3: Apply inference preprocessing transforms
# batch = preprocess(img).unsqueeze(0)

# print(time.time()-t0)
# # Step 4: Use the model and print the predicted category
# # prediction = model(batch).squeeze(0).softmax(0)
# # output = model(data)
# output = model(batch)
# print(time.time()-t0)
# # class_id = prediction.argmax().item()
# # score = prediction[class_id].item()
# # category_name = weights.meta["categories"][class_id]
# # print(f"{category_name}: {100 * score:.1f}%")
# # print(time.time()-t0)