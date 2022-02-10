from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack
from tqdm import tqdm
from utils.tools import EarlyStopping, adjust_learning_rate, StandardScaler
from utils.metrics import metric
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import wandb
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time

import warnings

warnings.filterwarnings('ignore')
wandb.init(project='Informer-Train', entity='vampire_acen')


def new_file(dir):
    file_lists = os.listdir(dir)
    if file_lists == []:
        return 'train_0'
    else:
        file_lists.sort(key=lambda x: os.path.getmtime((dir + "/" + x)))

        return file_lists[-1].split('_')[0] + '_' + str(int(file_lists[-1].split('_')[-1]) + 1)


file_name = new_file('./runs')


class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'informer': Informer,
            'informerstack': InformerStack,
        }
        if self.args.model == 'informer' or self.args.model == 'informerstack':
            e_layers = self.args.e_layers if self.args.model == 'informer' else self.args.s_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in,
                self.args.c_out,
                self.args.seq_len,
                self.args.label_len,
                self.args.pred_len,
                self.args.factor,
                self.args.d_model,
                self.args.n_heads,
                e_layers,  # self.args.e_layers,
                self.args.d_layers,
                self.args.d_ff,
                self.args.dropout,
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ETTh1': Dataset_ETT_hour,
            'ETTh2': Dataset_ETT_hour,
            'ETTm1': Dataset_ETT_minute,
            'ETTm2': Dataset_ETT_minute,
            'WTH': Dataset_Custom,
            'ECL': Dataset_Custom,
            'Solar': Dataset_Custom,
            'ZFDH': Dataset_Custom,
            'GSLL': Dataset_Custom,
            'RML': Dataset_Custom,
            'SMB': Dataset_Custom,
            'ZQWD': Dataset_Custom,
            'ZZQYL': Dataset_Custom,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed != 'timeF' else 1

        if flag == 'test':
            shuffle_flag = False;
            drop_last = True;
            batch_size = args.batch_size;
            freq = args.freq
        elif flag == 'pred':
            shuffle_flag = False;
            drop_last = False;
            batch_size = 1;
            freq = args.detail_freq
            Pred_Data = Dataset_Pred
        else:
            shuffle_flag = True;
            drop_last = True;
            batch_size = args.batch_size;
            freq = args.freq
        if flag == 'pred':
            data_set = Pred_Data(
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                size=[args.seq_len, args.label_len, args.pred_len],
                features=args.features,
                target=args.target,
                inverse=args.inverse,
                timeenc=timeenc,
                freq=freq,
                cols=args.cols,
                pred_data_path=args.pred_data_path,
                pred_result_path=args.pred_result_path,
                pred_data=args.pred_data
            )
        else:
            data_set = Data(
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                size=[args.seq_len, args.label_len, args.pred_len],
                features=args.features,
                target=args.target,
                inverse=args.inverse,
                timeenc=timeenc,
                freq=freq,
                cols=args.cols
            )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            pred, true = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        total_iter = 0
        config = wandb.config
        config.learning_rate = 0.001
        with SummaryWriter(log_dir='runs/' + file_name) as writer:
            for epoch in range(self.args.train_epochs):
                iter_count = 0
                train_loss = []

                self.model.train()

                with tqdm(iterable=train_loader) as bar:
                    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                        iter_count += 1
                        total_iter += 1
                        bar.set_description_str(f"【Train epoch {epoch + 1}】")
                        model_optim.zero_grad()
                        pred, true = self._process_one_batch(
                            train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                        loss = criterion(pred, true)
                        train_loss.append(loss.item())
                        writer.add_scalar('train loss iter', loss, total_iter)
                        writer.add_scalar('train epoch', epoch, epoch)
                        wandb.log({'train loss iter': loss})
                        wandb.log({'train epoch': epoch})
                        # if (i+1) % 100==0:
                        #     print()
                        #     print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        #     speed = (time.time()-time_now)/iter_count
                        #     left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                        #     print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        #     iter_count = 0
                        #     time_now = time.time()
                        bar.set_postfix_str(f"Train Loss:{loss.item():.6f}")

                        bar.update()
                        if self.args.use_amp:
                            scaler.scale(loss).backward()
                            scaler.step(model_optim)
                            scaler.update()
                        else:
                            loss.backward()
                            model_optim.step()

                    # print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
                    train_loss = np.average(train_loss)
                    vali_loss = self.vali(vali_data, vali_loader, criterion)
                    test_loss = self.vali(test_data, test_loader, criterion)
                    writer.add_scalar('train loss epoch', train_loss, epoch)
                    writer.add_scalar('vali loss epoch', vali_loss, epoch)
                    writer.add_scalar('test loss epoch', test_loss, epoch)
                    wandb.log({'train loss epoch': train_loss})
                    wandb.log({'vali loss epoch': vali_loss})
                    wandb.log({'test loss epoch': test_loss})
                    # bar.set_postfix_str(f"Train Loss:{train_loss:.6f},Vali Loss:{vali_loss:.6f},Test Loss:{test_loss:.6f}")
                    print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                        epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                    # bar.update()
                    early_stopping(vali_loss, self.model, path)
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break

                    adjust_learning_rate(model_optim, epoch + 1, self.args)
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')

        self.model.eval()

        preds = []
        trues = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(StandardScaler.inverse_transform(test_data.scaler, pred).detach().cpu().numpy())
            trues.append(StandardScaler.inverse_transform(test_data.scaler, true).detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

    def predict(self, setting, load=False):

        if load:
            if self.args.use_gpu or self.args.use_multi_gpu:
                path = os.path.join(self.args.checkpoints, setting)
                best_model_path = path + '/' + 'checkpoint.pth'
                self.model.load_state_dict(torch.load(best_model_path))
            else:
                path = os.path.join(self.args.checkpoints, setting)
                best_model_path = path + '/' + 'checkpoint.pth'
                # self.model.load_state_dict(torch.load(best_model_path),False)
                # original saved file with DataParallel
                state_dict = torch.load(best_model_path)
                # create new OrderedDict that does not contain `module.`
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                # load params
                self.model.load_state_dict(new_state_dict)

        self.model.eval()

        pred_data, pred_loader = self._get_data(flag='pred')

        preds = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
            pred, true = self._process_one_batch(
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(StandardScaler.inverse_transform(pred_data.scaler, pred).detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './pred/result/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        pred_name = self.args.pred_data.split('_')[-1]
        np.save(folder_path + 'pred_' + pred_name, preds)

        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        if self.args.padding == 0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding == 1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            wandb.watch(self.model)
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        # if self.args.graph:
        #     self.args.graph = False
        #     writer = SummaryWriter(log_dir='runs/'+file_name)
        #     model = self.model
        #     writer.add_graph(model, (batch_x, batch_x_mark, dec_inp, batch_y_mark))
        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)
        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        return outputs, batch_y
