# -*- coding: utf-8 -*-
import argparse

parser = argparse.ArgumentParser()

# 1. 训练超参数
parser.add_argument('--epoch', type=int, default=300, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='every n epochs decay learning rate')

# 2. 权重与断点 (关键：增加了 resume 参数)
parser.add_argument('--load_rgb', type=str, default='./mobilenet_v3_large-5c1a4163.pth', help='train from checkpoints')
parser.add_argument('--load_depth', type=str, default='./mobilevit_s.pt', help='train from checkpoints')
parser.add_argument('--resume', type=str, default=None, help='path to checkpoint for resume')
parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')

# 3. 训练数据集路径
parser.add_argument('--rgb_root', type=str, default='./train_dut/train_dut/train_images/', help='the training rgb images root')
parser.add_argument('--depth_root', type=str, default='./train_dut/train_dut/train_depth/', help='the training depth images root')
parser.add_argument('--gt_root', type=str, default='./train_dut/train_dut/train_masks/', help='the training gt images root')
parser.add_argument('--edge_root', type=str, default='./train_dut/train_dut/edge/', help='the training edge images root')

# 4. 测试与保存路径
# ⚠️ 【修改点】：将原先的 SGTID_v1 修正为 SFCINet，确保训练生成的权重保存在正确的文件夹中
parser.add_argument('--test_path', type=str, default='./test_datasets/test_datasets/', help='the test datasets root')
parser.add_argument('--save_path', type=str, default='./cpts/SFCINet/', help='the path to save models and logs')

opt = parser.parse_args()