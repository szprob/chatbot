import os
import pickle

import numpy as np

# import optim2
import torch
import tqdm


def save_file(f, path):
    if os.path.exists(path):
        os.remove(path)
    file = open(path, "wb")
    pickle.dump(f, file)
    file.close()


def _rule(epoch, warmup_steps=1000, down_steps=1e6):
    if down_steps < 10 * warmup_steps:
        down_steps = 10 * warmup_steps
    if epoch < warmup_steps:
        lamda = 8 * epoch / warmup_steps
    elif epoch < 2 * warmup_steps:
        lamda = 8 - 7 * (epoch - warmup_steps) / warmup_steps
    elif epoch < down_steps:
        lamda = 1.4 - (epoch - 2 * warmup_steps) / down_steps
    else:
        lamda = 0.4
    return lamda


class Trainer:
    def __init__(
        self,
        train_loader,
        log_freq: int = 100,
        save_freq: int = 3000,
        opt_freq=1,
        device="cuda:0",
    ):
        self.device = device
        self.logs = []
        # Setting the train and test data loader
        self.train_loader = train_loader
        # Setting the Adam optimizer with hyper-param
        # self.opt = optim2.Optimizer(self.model.parameters())
        # self.opt = optim.make_optimizer(self.model)

        self.log_freq = log_freq
        self.save_freq = save_freq
        self.opt_freq = opt_freq
        self.total_loss = []
        self.iter_num = 0

    def log_save(self, iter_num, loss_num, file_path, model):

        if (iter_num + 0) % self.log_freq == 0:
            self.total_loss.append(loss_num)
            log = (
                f"iter: {self.iter_num}",
                f"avg_loss : {round(np.mean(self.total_loss[-50:]),3)}",
                f"loss : {round(loss_num,3)}",
            )
            self.logs.append(log)
            print(log)

        if (iter_num + 0) % self.save_freq == 0:
            self.save((iter_num + 0), file_path, model)

    def train(self, model, file_path="/data/home/ze.song/models/sa", max_num=1e6):
        # self.loss_fct = CoralLoss()
        self.opt = torch.optim.AdamW(
            model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.001
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda=_rule)

        self.iteration(
            model,
            file_path,
            max_num=max_num,
        )

    def iteration(self, model, file_path=".", max_num=1e6):
        """
        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """

        data_iter = tqdm.tqdm(self.train_loader)
        iter_num = 0
        for data in data_iter:
            if iter_num > max_num:
                print("training finished!")
                break
            data = [d.to(self.device) for d in data]
            out2, out3, out5 = model(data[0], data[2], data[4])
            loss2 = self.loss_fct(out2, data[1].view(-1, 1))
            loss3 = self.loss_fct(out3, data[3])
            loss5 = self.loss_fct(out5, data[5])
            loss = 0.7 * loss2 + 0.4 * loss3 + loss5
            loss = loss / self.opt_freq

            loss.backward()
            loss_num = loss.item()

            if (iter_num + 0) % self.opt_freq == 0:

                torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
                self.opt.step()
                self.opt.zero_grad()
                self.scheduler.step()

            if self.device.index == 0:
                self.log_save(iter_num, loss_num, file_path, model)
            iter_num += 1

    def save(self, step, file_path="/data/home/ze.song/models/gpt", model=None):
        """
        Saving the current BERT model on file_path
        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        if not os.path.exists(file_path):
            os.mkdir(file_path)

        save_file(self.logs, file_path + "/logs.pkl")

        output_path = file_path + f"/model_{step}.pkl"
        torch.save(model.module.state_dict(), output_path)
        his_path = file_path + f"/model_{step-10*self.save_freq}.pkl"
        if os.path.exists(his_path):
            os.remove(his_path)
        return output_path
