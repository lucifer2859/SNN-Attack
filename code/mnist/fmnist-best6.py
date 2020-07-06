import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
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
'''
94.81

3层全连接+Dropout，Dropout增大到0.7

去掉阈值裕量

boost

自动化所baseline 92.07
'''

class Net(nn.Module):
    def __init__(self, v_threshold=1.0, v_reset=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            neuron.PLIFNode(v_threshold=v_threshold, v_reset=v_reset),
            nn.MaxPool2d(2, 2),  # 14 * 14

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset),
            nn.MaxPool2d(2, 2)  # 7 * 7

        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            layer.Dropout(0.7),
            nn.Linear(128 * 7 * 7, 128 * 3 * 3, bias=False),
            neuron.PLIFNode(v_threshold=v_threshold, v_reset=v_reset),
            layer.Dropout(0.7),
            nn.Linear(128 * 3 * 3, 128, bias=False),
            neuron.PLIFNode(v_threshold=v_threshold, v_reset=v_reset),
            nn.Linear(128, 100, bias=False),
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
    T = 50
    train_epoch = 99999999
    log_dir = input('输入保存tensorboard日志文件的位置，例如“./”  ')

    # 初始化数据加载器
    train_data_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.FashionMNIST(
            root=dataset_dir,
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)
    test_data_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.FashionMNIST(
            root=dataset_dir,
            train=False,
            transform=torchvision.transforms.ToTensor(),
            download=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False)

    writer = SummaryWriter(log_dir)

    # 初始化网络
    net = Net().to(device)
    print(net)
    # 使用Adam优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    train_times = 0
    max_test_acccuracy = 0
    for _ in range(train_epoch):
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

            # 损失函数为输出层神经元的脉冲发放频率，与真实类别的交叉熵
            # 这样的损失函数会使，当类别i输入时，输出层中第i个神经元的脉冲发放频率趋近1，而其他神经元的脉冲发放频率趋近0
            loss = F.mse_loss(out_spikes_counter_frequency, F.one_hot(label, 10).float())
            loss.backward()
            optimizer.step()
            # 优化一次参数后，需要重置网络的状态，因为SNN的神经元是有“记忆”的
            net.reset_()

            # 正确率的计算方法如下。认为输出层中脉冲发放频率最大的神经元的下标i是分类结果
            correct_rate = (out_spikes_counter_frequency.max(1)[1] == label).float().mean().item()
            writer.add_scalar('train_correct_rate', correct_rate, train_times)
            if train_times % 1024 == 0:
                # print(device, dataset_dir, batch_size, learning_rate, T, train_epoch, log_dir, max_test_acccuracy)
                print(sys.argv, 'train_times', train_times, 'train_correct_rate', correct_rate)
            train_times += 1
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
            writer.add_scalar('test_correct_rate', test_accuracy, train_times)
            if max_test_acccuracy < test_accuracy:
                max_test_acccuracy = test_accuracy
                torch.save(net, log_dir + '/net.pkl')

if __name__ == '__main__':
    main()




