import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import sys
sys.path.append('.')
import SpikingFlow.softbp.neuron as neuron
import SpikingFlow.softbp.layer as layer
from torch.utils.tensorboard import SummaryWriter
import SpikingFlow.softbp.soft_pulse_function as soft_pulse_function
import SpikingFlow.softbp.functional as functional
import readline
import numpy as np
import tqdm
import os
'''
和c18-aug相同，但通道数均为256，且dropout为0.5
'''

class Net(nn.Module):
    def __init__(self, v_threshold=1.0, v_reset=0.0):
        super().__init__()

        self.train_times = 0
        self.max_test_acccuracy = 0

        self.conv = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset),

            nn.MaxPool2d(2, 2),  # 16 * 16

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset),
            nn.MaxPool2d(2, 2)  # 8 * 8

        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            layer.Dropout(0.5),
            nn.Linear(256 * 8 * 8, 128 * 4 * 4, bias=False),
            neuron.PLIFNode(v_threshold=v_threshold, v_reset=v_reset),
            nn.Linear(128 * 4 * 4, 100, bias=False),
            neuron.PLIFNode(v_threshold=v_threshold, v_reset=v_reset)
        )
        self.boost = nn.AvgPool1d(10, 10)

    def forward(self, x):
        return self.boost(self.fc(self.conv(x)).unsqueeze(1)).squeeze()

    def reset_(self):
        for item in self.modules():
            if hasattr(item, 'reset'):
                item.reset()
def main():
    device = input('输入运行的设备，例如“cpu”或“cuda:0”  ')
    dataset_dir = './dataset'
    # dataset_dir = '/datasets/MLG/wfang/fmnist'
    # batch_size = int(input('输入batch_size，例如“64”  '))
    batch_size = 128
    # learning_rate = float(input('输入学习率，例如“1e-3”  '))
    learning_rate = 1e-4
    # T = int(input('输入仿真时长，例如“50”  '))
    T = 8
    log_dir = input('输入保存tensorboard日志文件的位置，例如“./”  ')

    # 初始化数据加载器
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_data_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.CIFAR10(
            root=dataset_dir,
            train=True,
            transform=transform_train,
            download=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)
    test_data_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.CIFAR10(
            root=dataset_dir,
            train=False,
            transform=transform_test,
            download=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False)

    # net = torch.load(log_dir + '/net.pkl', map_location=device)
    # save_test_result(net, train_data_loader, T, device, log_dir)
    # exit()

    writer = SummaryWriter(log_dir)



    # 初始化网络
    if os.path.exists(log_dir + '/net.pkl'):
        net = torch.load(log_dir + '/net.pkl', map_location=device)
        print(net.train_times, net.max_test_acccuracy)
    else:
        net = Net().to(device)
    print(net)
    # 使用Adam优化器
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)


    while True:
        net.train()
        for img, label in train_data_loader:
            img = img.to(device)
            label = label.to(device)
            optimizer.zero_grad()

            # 运行T个时长，out_spikes_counter是shape=[batch_size, 10]的tensor
            # 记录整个仿真时长内，输出层的10个神经元的脉冲发放次数
            for t in range(T):
                if t == 0:
                    out_spikes_counter = net(img)
                else:
                    out_spikes_counter += net(img)

            # out_spikes_counter / T 得到输出层10个神经元在仿真时长内的脉冲发放频率
            out_spikes_counter_frequency = out_spikes_counter / T

            loss = F.mse_loss(out_spikes_counter_frequency, F.one_hot(label, 10).float())
            loss.backward()
            optimizer.step()
            net.reset_()

            # 正确率的计算方法如下。认为输出层中脉冲发放频率最大的神经元的下标i是分类结果
            correct_rate = (out_spikes_counter_frequency.max(1)[1] == label).float().mean().item()
            if net.train_times % 128 == 0:
                writer.add_scalar('train_correct_rate', correct_rate, net.train_times)
                writer.add_scalar('train_loss', loss.item(), net.train_times)

            if net.train_times % 1024 == 0:
                print(device, dataset_dir, batch_size, learning_rate, T, log_dir, net.max_test_acccuracy)
                print(sys.argv, 'train_times', net.train_times, 'train_correct_rate', correct_rate)
            net.train_times += 1
        net.eval()

        with torch.no_grad():
            # 每遍历一次全部数据集，就在测试集上测试一次
            test_sum = 0
            correct_sum = 0
            for img, label in test_data_loader:
                img = img.to(device)
                label = label.to(device)
                for t in range(T):
                    if t == 0:
                        out_spikes_counter = net(img)
                    else:
                        out_spikes_counter += net(img)

                correct_sum += (out_spikes_counter.max(1)[1] == label).float().sum().item()
                test_sum += label.numel()
                net.reset_()
            test_accuracy = correct_sum / test_sum
            writer.add_scalar('test_correct_rate', test_accuracy, net.train_times)
            if net.max_test_acccuracy < test_accuracy:
                try:
                    net.max_test_acccuracy = test_accuracy
                    torch.save(net, log_dir + '/net.pkl')
                except KeyboardInterrupt:
                    net.max_test_acccuracy = test_accuracy
                    torch.save(net, log_dir + '/net.pkl')
                    exit()



if __name__ == '__main__':
    main()




