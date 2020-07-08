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

def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    t = torch.clamp(t, 0.0, 1.0)
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

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
    T = 50
    phase = 'BIM'

    torch.cuda.empty_cache()

    encoder = encoding.PoissonEncoder()

    if phase == 'BIM':
        # model_path = input('输入模型文件路径，例如“./model.pth”  ')
        # iter_num = int(input('输入对抗攻击的迭代次数，例如“25”  '))
        # eta = float(input('输入对抗攻击学习率，例如“0.05”  '))
        # attack_type = input('输入攻击类型，例如“UT/T”  ')
        # clip_flag = bool(input('输入是否使用截断，例如“True/False”  '))
        # clip_eps = float(input('输入截断eps，例如“0.01”  '))

        model_path = './models/mnist_spike_v1.pth'
        iter_num = 100
        eta = 0.02
        attack_type = 'T'

        clip_flag = True
        clip_eps = 0.4

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

                    rate = torch.zeros_like(spike).to(device)

                    for spike in spike_train:
                        rate += spike.grad.data

                    img_grad = torch.sign(rate)

                    if clip_flag:
                        img = clip_by_tensor(img + eta * img_grad, img_ori - clip_eps, img_ori + clip_eps)
                    else:
                        img = torch.clamp(img + eta * img_grad, 0.0, 1.0)

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

                        # loss = F.mse_loss(out_spikes_counter_frequency, F.one_hot(target_label, class_num).float())
                        loss = F.cross_entropy(out_spikes_counter_frequency, target_label)
                        
                        loss.backward()

                        rate = torch.zeros_like(spike).to(device)

                        for spike in spike_train:
                            rate += spike.grad.data

                        img_grad = torch.sign(rate)

                        if clip_flag:
                            img = clip_by_tensor(img - eta * img_grad, img_ori - clip_eps, img_ori + clip_eps)
                        else:
                            img = torch.clamp(img - eta * img_grad, 0.0, 1.0)

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