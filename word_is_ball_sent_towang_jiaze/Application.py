# _*_ coding:utf-8 _*_ 
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import sys

sys.path.append('./')
from data.AceData import AceData
from core.Ace3d_model import MaoXia3DNet

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


class App:
    def __init__(self, args):
        self.args = args
        self.prefix_dir = os.path.join("./models", "ballnet_" + args.train_id)
        self.gpus = args.gpus
        self.device = self.find_device()
        self.model = self.get_model()
        self.Datasets = self.get_datasets(args)
        self.Loss = self.get_lossFunction()
        self.opt = self.get_optim()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, "min", patience=args.lr_step,
                                                                    min_lr=0.00001, verbose=True)
        t = time.localtime()
        self.writer = SummaryWriter(
            log_dir=os.path.join(self.prefix_dir, "log_tensorboard",
                                 "%s_%s_%s" % (str(t.tm_mon), str(t.tm_mday), str(t.tm_hour))))
        print("what is scheduler look like: ", self.scheduler)

    def get_datasets(self,args):
        train_set = AceData(file=args.data_path, device=self.device)
        valid_set = AceData(file=args.val_path, device=self.device)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        return [train_loader, valid_loader]

    def get_model(self):
        if self.args.pretrain:
            model_path = os.path.join(self.prefix_dir, "model_" + str(self.args.pretrain_epoch) + ".pth")
            model = torch.load(model_path)
        else:
            if self.args.model_name == "3d_ace":
                model = MaoXia3DNet(num_classes=self.args.num_class)
            else:
                print("there is no model load, please check your code!")
        if len(self.gpus) > 1:
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
        model.to(self.device)
        return model

    def find_device(self):
        print("we wanna get device in computer ! ")
        USE_CUDA = torch.cuda.is_available()
        print(USE_CUDA)
        device = torch.device("cuda:0" if USE_CUDA else "cpu")
        # device = torch.device("cpu")
        print(device)
        return device

    def get_lossFunction(self):
        # add others loss function on here,Now, just use jiaochashang
        loss = nn.CrossEntropyLoss()
        loss.to(self.device)
        # loss = Cognitive(self.device)
        return loss

    def get_optim(self):
        # opt = optim.Adam(filter(lambda p:p.requires_grad, self.model.parameters()),lr = self.args.lr)
        opt = optim.Adam(self.model.parameters(), lr=self.args.lr)
        return opt

    def save_log(self, tags, values, n_iter):
        # dir_path = os.path.join('logs',dir_name)
        for tag, value in zip(tags, values):
            self.writer.add_scalar(tag, value, n_iter)
            self.writer.add_text(tag, str(value), n_iter)

    def write_f1_recall_precision(self, values, n_iter):
        loss, f1, recall, precision, _ = values
        with open(os.path.join(self.prefix_dir, "loss_f1_recall_precision.txt"), "a+") as f:
            f.write(str(loss) + "," + str(f1) + "," + str(recall) + "," + str(precision) + "," + str(n_iter) + "\n")

    def compare_numpy(self, a, b):
        # print(a.shape, b.shape)
        # print("[", a[0].detach().cpu().numpy()," | ", b[0].detach().cpu().numpy(),end="]")
        # print(a.shape[0])
        c = np.zeros(a.shape[0])
        for i in range(a.shape[0]):
            loss = torch.pow((a[i][0] - b[i][0]), 2) + torch.pow((a[i][1] - b[i][1]), 2)
            if loss < 4:
                c[i] = True
            else:
                c[i] = False
        return c

    def backwark_grid(self):
        for name, params in self.model.named_parameters():
            print(name, params)
            if type(params) == "NoneType":
                print(name)
                print("params is None! ")
                break
            print('-->name:', name, '-->grad_requirs:', params.requires_grad, '-->grad_value:', params.grad, "--> data",
                  params.data)
        print("----------------------------------------------------------------------------------------------")

    def calculat_roc(self, net_output, label):
        # print(net_output.shape)
        y_pre = torch.argmax(net_output, dim=1)
        y_pre = y_pre.detach().cpu().numpy()
        label = label.detach().cpu().numpy()
        precision = precision_score(label, y_pre, average='macro')
        recall = recall_score(label, y_pre, average='macro')
        f1 = f1_score(label, y_pre, average='macro')
        return f1, recall, precision

    def update(self, data):
        img, label = data['item'].to(self.device), data['label'].to(self.device)
        # print("img shape :", img.shape)
        self.opt.zero_grad()
        output = self.model(img)
        loss = self.Loss(output, label)
        loss.backward()
        # check the info of  backward 
        # self.backwark_grid()
        self.opt.step()
        lr = self.opt.state_dict()['param_groups'][0]['lr']
        # print("learning rate :", lr)
        f1_score, recall, precision = self.calculat_roc(output, label)
        return [loss.cpu().detach().numpy(), f1_score, recall, precision, lr]
        # return [loss.cpu().detach().numpy(),0,0,0,lr]

    def forward(self, data):
        img, label = data['item'].to(self.device), data['label'].to(self.device)

        output = self.model(img)
        loss = self.Loss(output, label)
        f1_score, recall, precision = self.calculat_roc(output, label)
        return [loss.cpu().detach().numpy(), f1_score, recall, precision]

    def save_model(self, epoch):
        torch.save(self.model.state_dict(),
                   os.path.join(self.prefix_dir, "model_" + str(epoch) + ".pth"))

    def validation(self, valid_loader, n_iter):
        print("start validation -->")
        val_loss, val_f1, val_recall, val_precision = 0., 0., 0., 0.
        step = 0
        for v_batch_i, v_data in enumerate(valid_loader):
            
            if v_data['item'].shape[0] <= 1:
                print("get a error shape of item data:", data['item'].shape)
                continue
            v_values = self.forward(v_data)
            val_loss += v_values[0]
            val_f1 += v_values[1]
            val_recall += v_values[2]
            val_precision += v_values[3]
            step += 1
        all_rate = [val_loss / step, val_f1 / step, val_recall / step, val_precision / step]
        self.save_log(['validation/loss', 'validation/f1_score', 'validation/recall', 'validation/precision'], all_rate,
                      n_iter)
        print("[validation]: n_iter[%d],---> loss[%.4f], precision[%.4f], recall[%.4f], f1_score[%.4f] " % (
        n_iter, all_rate[0], all_rate[-1], all_rate[3], all_rate[2]))
        self.save_model(n_iter)
        return all_rate[0]

    def train(self):
        train_loader, valid_loader = self.Datasets
        n_iter = 0
        for epoch in range(self.args.epochs):
            train_loss, train_f1, train_recall, train_precision = 0., 0., 0., 0.
            step = 0
            
            for batch_i, data in enumerate(train_loader):
                # print("type:", type(data['item']))
                if data['item'] is None:
                    print("get no data:")
                    continue
                if data['item'].shape[0] <= 1:
                    print("get a error shape of item data:", data['item'].shape)
                    continue

                values = self.update(data)
                train_loss += values[0]
                train_f1 += values[1]
                train_recall += values[2]
                train_precision += values[3]
                step += 1
                n_iter += data['label'].shape[0]
                print(
                    "[train]:epoch[%d], n_iter[%d],---> loss[%.4f], precision[%.4f], recall[%.4f], f1_score[%.4f] " % (
                    epoch, n_iter, values[0], values[-2], values[2], values[1]))

                if (batch_i) % 200 == 0:
                   average_loss = self.validation(valid_loader, n_iter)
                   self.scheduler.step(average_loss)
                all_rate = [train_loss / step, train_f1 / step, train_recall / step, train_precision / step, values[-1]]
                self.write_f1_recall_precision(all_rate, n_iter)
                self.save_log(
                    ['train/loss', 'train/f1_score', 'train/recall', 'train/precision', 'train/learning_rate'],
                    all_rate, n_iter)
            self.save_model(epoch)
            # validation
