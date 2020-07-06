import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
import sys
sys.path.append('../')
import SpikingFlow.softbp.neuron as neuron
import SpikingFlow.softbp.layer as layer
import SpikingFlow.encoding as encoding
import SpikingFlow.softbp.soft_pulse_function as soft_pulse_function
from torch.utils.tensorboard import SummaryWriter
import readline
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

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
    # device = input('输入运行的设备，例如“cpu”或“cuda:0”  ')
    # dataset_dir = input('输入保存MNIST数据集的位置，例如“./”  ')
    # class_num = int(input('输入class_num，例如“10”  '))
    # lr = float(input('输入学习率，例如“1e-3”  '))
    # T = int(input('输入仿真时长，例如“50”  ')) 
    # phase = input('输入算法阶段，例如“train/BIM”  ')

    device = 'cuda:1'
    dataset_dir = '../../dataset/'
    class_num = 10
    lr = 1e-4
    T = 50
    phase = 'BIM'

    torch.cuda.empty_cache()

    encoder = encoding.PoissonEncoder()

    if phase == 'train':
        # model_dir = input('输入保存模型文件的位置，例如“./”  ')
        # batch_size = int(input('输入batch_size，例如“64”  '))
        # train_epoch = int(input('输入训练轮数，即遍历训练集的次数，例如“100”  '))
        # log_dir = input('输入保存tensorboard日志文件的位置，例如“./”  ')

        model_dir = './models/'
        batch_size = 64
        train_epoch = 9999999
        log_dir = './logs/'

        writer = SummaryWriter(log_dir)

        train_data_loader = torch.utils.data.DataLoader(
            dataset=torchvision.datasets.MNIST(
                root=dataset_dir,
                train=True,
                transform=torchvision.transforms.ToTensor(),
                download=True),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True)

        test_data_loader = torch.utils.data.DataLoader(
            dataset=torchvision.datasets.MNIST(
                root=dataset_dir,
                train=False,
                transform=torchvision.transforms.ToTensor(),
                download=True),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False)

        net = Net().to(device)
        
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        
        train_times = 0
        best_epoch = 0
        max_correct_sum = 0

        for epoch in range(1, train_epoch + 1):
            net.train()
            for X, y in train_data_loader:
                img, label = X.to(device), y.to(device)
                
                optimizer.zero_grad()

                for t in range(T):
                    if t == 0:
                        out_spikes_counter = net(encoder(img).float())
                    else:
                        out_spikes_counter += net(encoder(img).float())

                out_spikes_counter_frequency = out_spikes_counter / T

                loss =  F.mse_loss(out_spikes_counter_frequency, F.one_hot(label, class_num).float())
                # loss = F.cross_entropy(out_spikes_counter_frequency, label)
                
                loss.backward()
                optimizer.step()
                
                net.reset_()

                correct_rate = (out_spikes_counter_frequency.max(1)[1] == label).float().mean().item()
                writer.add_scalar('train_correct_rate', correct_rate, train_times)
                # if train_times % 1024 == 0:
                #     print(device, dataset_dir, batch_size, lr, T, train_epoch, log_dir)
                #     print(sys.argv, 'train_times', train_times, 'train_correct_rate', correct_rate)
                train_times += 1

            net.eval()
            
            with torch.no_grad():
                test_sum = 0
                correct_sum = 0
                for X, y in test_data_loader:
                    img, label = X.to(device), y.to(device)
                    for t in range(T):
                        if t == 0:
                            out_spikes_counter = net(encoder(img).float())
                        else:
                            out_spikes_counter = net(encoder(img).float())

                    correct_sum += (out_spikes_counter.max(1)[1] == label).float().sum().item()
                    test_sum += label.numel()
                    net.reset_()

                writer.add_scalar('test_correct_rate', correct_sum / test_sum, train_times)

                print('epoch', epoch, 'test_correct_rate', correct_sum / test_sum)

                if correct_sum > max_correct_sum:
                    max_correct_sum = correct_sum
                    torch.save(net.state_dict(), model_dir + 'spike_best_%d.pth' % (epoch))
                    if best_epoch > 0:
                        os.system('rm %sspike_best_%d.pth' % (model_dir, best_epoch))
                    best_epoch = epoch

    elif phase == 'BIM':
        # model_path = input('输入模型文件路径，例如“./model.pth”  ')
        # iter_num = int(input('输入对抗攻击的迭代次数，例如“25”  '))
        # gamma = float(input('输入GT的采样因子，例如“0.05”  '))
        # perturbation = float(input('输入扰动幅度，例如“4.0”  '))
        # attack_type = input('输入攻击类型，例如“UT/T”  ')

        model_path = './models/mnist_spike_v1.pth'
        gamma = 0.05
        iter_num = 50
        perturbation = 3.1
        attack_type = 'T'

        test_data_loader = torch.utils.data.DataLoader(
            dataset=torchvision.datasets.MNIST(
                root=dataset_dir,
                train=False,
                transform=torchvision.transforms.ToTensor(),
                download=True),
            batch_size=1,
            shuffle=False,
            drop_last=False)

        net = Net().to(device)
        net.load_state_dict(torch.load(model_path))

        mean_p = 0.0
        test_sum = 0
        success_sum = 0

        if attack_type == 'UT':
            for X, y in test_data_loader:
                img, label = X.to(device), y.to(device)
                img_ori = torch.rand_like(img).copy_(img)

                test_sum += 1

                print('Img %d' % test_sum)

                net.train()

                for it in range(iter_num):
                    spike_train = []

                    for t in range(T):
                        if t == 0:
                            spike = encoder(img).float()
                            spike.requires_grad = True
                            spike_train.append(spike)
                            out_spikes_counter = net(spike).unsqueeze(0)
                        else:
                            spike = encoder(img).float()
                            spike.requires_grad = True
                            spike_train.append(spike)
                            out_spikes_counter += net(spike).unsqueeze(0)

                    out_spikes_counter_frequency = out_spikes_counter / T

                    # loss = F.mse_loss(out_spikes_counter_frequency, F.one_hot(label, class_num).float())
                    loss = F.cross_entropy(out_spikes_counter_frequency, label)

                    loss.backward()

                    ik = torch.zeros_like(spike).to(device)

                    for spike in spike_train:
                        if torch.max(torch.abs(spike.grad.data)) > 1e-32:
                            # print('G2S Converter')

                            grad_sign = torch.sign(spike.grad.data)
                            grad_abs = torch.abs(spike.grad.data)
                            grad_norm = (grad_abs - torch.min(grad_abs)) / (torch.max(grad_abs) - torch.min(grad_abs))
                            grad_mask = torch.bernoulli(grad_norm)
                            G2S = grad_sign * grad_mask
                            G2S_trans = torch.clamp(G2S + spike, 0.0, 1.0) - spike

                            ik += G2S_trans

                        else:
                            # print('Gradient Trigger')

                            GT = torch.bernoulli(torch.ones_like(spike.grad.data) * gamma)
                            GT_trans = (GT.bool() ^ spike.bool()).float() - spike

                            ik += GT_trans

                    ik /= T

                    l2_norm = torch.norm(ik.view(ik.size()[0], -1), dim=1).item()
                    # print('Perturbation: %f' % l2_norm)

                    if l2_norm < perturbation:
                        img = torch.clamp(img + ik, 0.0, 1.0)

                        net.reset_()

                        for p in spike_train:
                            p.grad.data.zero_()
                        for p in net.parameters():
                            p.grad.data.zero_()

                    else:
                        net.reset_()

                        for p in spike_train:
                            p.grad.data.zero_()
                        for p in net.parameters():
                            p.grad.data.zero_()

                net.eval()

                with torch.no_grad():
                    img_diff = img - img_ori

                    l2_norm = torch.norm(img_diff.view(img_diff.size()[0], -1), dim=1).item()
                    print('Total Perturbation: %f' % l2_norm)

                    mean_p += l2_norm

                    for t in range(T):
                        if t == 0:
                            out_spikes_counter = net(encoder(img).float()).unsqueeze(0)
                        else:
                            out_spikes_counter += net(encoder(img).float()).unsqueeze(0)

                    out_spikes_counter_frequency = out_spikes_counter / T

                    attack_flag = (out_spikes_counter.max(1)[1] != label).float().sum().item()
                    success_sum += attack_flag

                    if attack_flag > 0.5:
                        print('Attack Success')
                    else:
                        print('Attack Failure')

                if test_sum >= 250:
                    mean_p /= 250
                    break 
        else:
            for X, y in test_data_loader:
                for i in range(1, class_num):
                    img, label = X.to(device), y.to(device)
                    img_ori = torch.rand_like(img).copy_(img)

                    target_label = (label + i) % class_num

                    test_sum += 1

                    net.train()

                    for it in range(iter_num):
                        spike_train = []

                        for t in range(T):
                            if t == 0:
                                spike = encoder(img).float()
                                spike.requires_grad = True
                                spike_train.append(spike)
                                out_spikes_counter = net(spike).unsqueeze(0)
                            else:
                                spike = encoder(img).float()
                                spike.requires_grad = True
                                spike_train.append(spike)
                                out_spikes_counter += net(spike).unsqueeze(0)

                        out_spikes_counter_frequency = out_spikes_counter / T

                        loss = F.mse_loss(out_spikes_counter_frequency, F.one_hot(target_label, class_num).float())
                        # loss = F.cross_entropy(out_spikes_counter_frequency, target_label)

                        loss.backward()

                        ik = torch.zeros_like(spike).to(device)

                        for spike in spike_train:
                            if torch.max(torch.abs(spike.grad.data)) > 1e-32:
                                # print('G2S Converter')

                                grad_sign = -torch.sign(spike.grad.data)
                                grad_abs = torch.abs(spike.grad.data)
                                grad_norm = (grad_abs - torch.min(grad_abs)) / (torch.max(grad_abs) - torch.min(grad_abs))
                                grad_mask = torch.bernoulli(grad_norm)
                                G2S = grad_sign * grad_mask
                                G2S_trans = torch.clamp(G2S + spike, 0.0, 1.0) - spike

                                ik += G2S_trans

                            else:
                                # print('Gradient Trigger')

                                GT = torch.bernoulli(torch.ones_like(spike.grad.data) * gamma)
                                GT_trans = (GT.bool() ^ spike.bool()).float() - spike

                                ik += GT_trans

                        ik /= T

                        l2_norm = torch.norm(ik.view(ik.size()[0], -1), dim=1).item()
                        # print('Perturbation: %f' % l2_norm)

                        if l2_norm < perturbation:
                            img = torch.clamp(img + ik, 0.0, 1.0)

                            net.reset_()

                            for p in spike_train:
                                p.grad.data.zero_()
                            for p in net.parameters():
                                p.grad.data.zero_()

                        else:
                            net.reset_()

                            for p in spike_train:
                                p.grad.data.zero_()
                            for p in net.parameters():
                                p.grad.data.zero_()

                    net.eval()

                    with torch.no_grad():
                        img_diff = img - img_ori

                        l2_norm = torch.norm(img_diff.view(img_diff.size()[0], -1), dim=1).item()
                        print('Total Perturbation: %f' % l2_norm)

                        mean_p += l2_norm

                        for t in range(T):
                            if t == 0:
                                out_spikes_counter = net(encoder(img).float()).unsqueeze(0)
                            else:
                                out_spikes_counter += net(encoder(img).float()).unsqueeze(0)

                        out_spikes_counter_frequency = out_spikes_counter / T

                        attack_flag = (out_spikes_counter.max(1)[1] == target_label).float().sum().item()
                        success_sum += attack_flag

                        if attack_flag > 0.5:
                            print('Attack Success')
                        else:
                            print('Attack Failure')

                        '''
                        samples = img.permute(0, 2, 3, 1).data.cpu().numpy()

                        im = np.repeat(samples[0], 3, axis=2)
                        im_path = 'demo/%d_to_%d.png' % (label.item(), target_label.item())
                        print(im_path)
                        print(out_spikes_counter_frequency)
                        plt.imsave(im_path, im)
                        '''

                if test_sum >= 270:
                    mean_p /= 270
                    break

        print('Mean Perturbation: %.2f' % mean_p)
        print('success_sum: %d' % success_sum)
        print('test_sum: %d' % test_sum)
        print('success_rate: %.2f%%' % (100 * success_sum / test_sum))  

if __name__ == '__main__':
    main()