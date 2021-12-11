# -*- encoding: utf-8 -*-
"""
@File    :   AceData.py    
@Contact :   mr.wangshuxian@qq.com

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/7/20 20:41   dlwang      1.0         毛虾数据集
"""
import cv2
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co
from torchvision import transforms
import numpy as np
import torch


class AceData(Dataset):

    def __init__(self, file, device):
        """
        :param file: 处理后的txt文件 每行一条记录 每条记录包括文件名（含路径）、帧数范围、 标签类别
        :param device:
        """
        self.data_list = self.get_list(file)
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.device = device

    def get_list(self, file):
        lines = None
        with open(file, 'r') as f:
            lines = f.readlines()
        return lines

    def __len__(self):
        data_length = len(self.data_list)
        return data_length

    def __getitem__(self, index):
        line = self.data_list[index]
        file_name = 'E:/syy/' + line.split(',')[0]
        frames_start = line.split(',')[1].split('-')[0]
        frames_stop = line.split(',')[1].split('-')[1]
        label = line.split(',')[2].split('\n')[0]
        try:
            with open(file_name, 'r')as f:
                f.close
        except Exception as e:
            print(e, file_name)
            batch_data = torch.randn(100,3,256,256)
            return {'item': batch_data, 'label': int(label)}

        capture = cv2.VideoCapture(file_name)
        # print(file_name)
        frames_num = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frames_start))  # set the start frame
        status, frame = capture.read()
        if (status) and (frame is not None):
            batch_data = self.transforms(frame)
            batch_data = batch_data.unsqueeze_(0)
        else:
            print("the first frame None!!! add a random start frame")
            batch_data = torch.randn(3,256,256)
            batch_data = batch_data.unsqueeze_(0)
        while (int(capture.get(cv2.CAP_PROP_POS_FRAMES)) < int(frames_stop)):
            is_ok,image = capture.read()
            if (image is None) or(not is_ok):
                capture.set(cv2.CAP_PROP_POS_FRAMES, capture.get(cv2.CAP_PROP_POS_FRAMES) + 1)
                if (int(frames_stop) + 1) < frames_num:
                    frames_stop = int(frames_stop) + 1
                    print("run frame: ", capture.get(cv2.CAP_PROP_POS_FRAMES))
                    continue
                else:
                    break
                
            image = self.transforms(image)
            image = image.unsqueeze_(0)
            # print(image.shape)
            if image.shape != torch.Size([1, 3, 256, 256]):
                print(image.shape)
                print("find a none image!!!")
                break
            batch_data = torch.cat((batch_data, image), dim=0)
            
        if batch_data.shape[0] != 100:
            print("add noise !!!!error get :", batch_data.shape)
            print(line)
            batch_data = torch.randn(100,3,256,256)
        return {'item': batch_data, 'label': int(label)}
