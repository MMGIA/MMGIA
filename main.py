import argparse
import pickle

import torch
import time

from dlg import *
import copy
from util.util import *
from util.plot_util import *

from torch.optim import Adam
import torch.nn as nn
import numpy as np
import math


def main(args):
    torch.backends.cudnn.enabled = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    start_time = time.time()

    save_path = 'result' + r'\expe_[{}]_{}'.format(args.experiment_name,
                                                   time.strftime('%m-%d-%H-%M-%S', time.localtime(start_time)))
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for arg in vars(args):
        print(format(arg, '<20'), format(str(getattr(args, arg)), '<'))  # str, arg_type
    save_args(args, save_path)

    dataset, scaler = init_dataset(args)
    for er in range(args.experiment_rounds):
        print("====== Repeat {} ======".format(er))
        # ========================= START TRAIN ==============================
        model = init_model(args,
                           model_name=args.model_name,
                           input_size=2,
                           hidden_size=args.hidden_size,
                           output_size=2)
        model = model.to(device)

        loss_fn = init_loss(args.model_name).to(device)

        # 记录
        gt_data_list = []
        gt_tim_list = []
        gt_label_list = []
        true_loss_list = []

        model.train()

        for iter in range(args.global_iterations):
            print("------Iteration {}------".format(iter))
            # 获取数据
            batch_size = args.batch_size
            gt_data_np = dataset[iter: iter + batch_size, [2, 3]]
            gt_tim_np = dataset[iter: iter + batch_size, [0]]
            gt_label_np = dataset[iter + batch_size, [2, 3]]

            # 记录
            gt_data_list.append(gt_data_np)
            gt_tim_list.append(gt_tim_np)
            gt_label_list.append(gt_label_np)

            if iter == 0:
                continue

            # 原任务训练
            true_loss = train_for_loss(args,
                                       model_name=args.model_name,
                                       model=model,
                                       loss_fn=loss_fn,
                                       batch_size=batch_size,
                                       gt_data_np=gt_data_np,
                                       gt_tim_np=gt_tim_np,
                                       gt_label_np=gt_label_np,
                                       global_iter=iter,
                                       dataset=dataset,
                                       device=device
                                       )

            true_loss_list.append(true_loss.item())
            dy_dx = torch.autograd.grad(true_loss, model.parameters(), retain_graph=True)

            # # =========================================== DP-SGD ============================================ #
            # if args.is_DP:
            #     for d in dy_dx:
            #         norm2 = torch.norm(d, p=2)
            #         d.data = d.data / max(1, norm2 / args.dp_C)
            #         mean = torch.zeros(d.shape)
            #         std = math.sqrt(2 * math.log(1.25 / args.dp_delta)) / args.dp_epsilon * args.dp_C
            #         noise = torch.normal(mean, std).to(device)
            #         d.data = d.data + noise
            # # ============================================================================================= #

            # =========================================== dlg attack =========================================== #
            if args.is_dlg == 1 and iter == 1:
                attack_record = dlg_attack(args,
                                           batch_size=args.batch_size,
                                           model=copy.deepcopy(model),
                                           true_dy_dx=dy_dx,
                                           dlg_attack_round=args.dlg_attack_rounds,
                                           dlg_iteration=args.dlg_iterations,
                                           dlg_lr=args.dlg_lr,
                                           global_iter=iter,
                                           model_name=args.model_name,
                                           gt_data_np=gt_data_np,
                                           gt_tim_np=gt_tim_np,
                                           gt_label_np=gt_label_np,
                                           save_path=save_path,
                                           device=device,
                                           dataset=dataset,
                                           er=er,
                                           )
            # ================================================================================================== #

            for server_param, grad_param in zip(model.parameters(), dy_dx):
                server_param.data = server_param.data - args.lr * grad_param.data.clone()

            print("Iter {} origin task: true_loss:{}".format(iter, true_loss))

        pickle.dump(gt_data_list, open(save_path + '/gt_data_list_er={}.pickle'.format(er), 'wb'))
        pickle.dump(gt_tim_list, open(save_path + '/gt_tim_list_er={}.pickle'.format(er), 'wb'))
        pickle.dump(gt_label_list, open(save_path + '/gt_label_list_er={}.pickle'.format(er), 'wb'))
        pickle.dump(true_loss_list, open(save_path + '/true_loss_list_er={}.pickle'.format(er), 'wb'))
        print("Finish Train!")

        # ====================== START TEST ==========================
        print("Start Test!")
        model.eval()
        test_data_list = []
        test_label_list = []
        test_pred_list = []
        for i in tqdm(range(2000, 2100)):
            batch_size = args.batch_size
            gt_data_np = dataset[i: i + batch_size, [2, 3]]
            gt_tim_np = dataset[i: i + batch_size, [0]]
            gt_label_np = dataset[i + batch_size, [2, 3]]
            test_data_list.append(gt_data_np)
            test_label_list.append(gt_label_np)
            gt_data = torch.from_numpy(gt_data_np.reshape(1, batch_size, 2).astype(np.float32)).to(device)
            pred = model(gt_data)
            test_pred_list.append(pred.cpu().detach().numpy())
        pickle.dump(test_data_list, open(save_path + '/test_data_list_er={}.pickle'.format(er), 'wb'))
        pickle.dump(test_label_list, open(save_path + '/test_label_list_er={}.pickle'.format(er), 'wb'))
        pickle.dump(test_pred_list, open(save_path + '/test_pred_list_er={}.pickle'.format(er), 'wb'))
        print("Finished Test!")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_name", type=str, default='test', help="the name of experiment")
    parser.add_argument("--dataset_name", type=str, default='nybc', help="dataset",
                        choices=['nybc', 'tokyoci', 'gowallaci'])
    parser.add_argument("--data_dir", type=str, default='data/mta_1706.csv', help="the address of a selected dataset")
    parser.add_argument("--usernum", type=int, default=1, help="user number")
    parser.add_argument("--batch_size", type=int, default=1, help="the size of data used in training")
    parser.add_argument("--model_name", type=str, default='LSTM', help="the model you use",
                        choices=['LSTM', 'PMF', 'DeepMove'])
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate for original task")
    parser.add_argument("--hidden_size", type=int, default=16, help="the hidden size of model")
    parser.add_argument("--experiment_rounds", type=int, default=100, help="the rounds of experiments")
    parser.add_argument("--global_iterations", type=int, default=2000, help="the global iterations for original task")

    # param for DLG
    parser.add_argument("--is_dlg", type=int, default=1, help="if use dlg")
    parser.add_argument("--dlg_attack_interval", type=int, default=5, help="the interval between two dlg attack")
    parser.add_argument("--dlg_attack_rounds", type=int, default=4, help="the rounds of dlg attack")
    parser.add_argument("--dlg_iterations", type=int, default=200, help="the iterations for dlg attack")
    parser.add_argument("--dlg_lr", type=float, default=0.05, help="learning rate for dlg attack")

    # param for DP-SGD
    parser.add_argument("--is_DP", type=int, default=0, help="if use DP")
    parser.add_argument("--dp_C", type=float, default=0.2, help="clipping threshold in DP")
    parser.add_argument("--dp_epsilon", type=float, default=8, help="epsilon in DP")
    parser.add_argument("--dp_delta", type=float, default=1e-5, help="delta in DP")

    # param for Geo-Indistinguishability
    parser.add_argument("--is_geo", type=int, default=0, help="if use geo")
    parser.add_argument("--geo_epsilon", type=float, default=8, help="epsilon in geo")

    # param for Geo-Graph-Indistinguishability
    parser.add_argument("--is_geogi", type=int, default=0, help="if use geogi")
    parser.add_argument("--geogi_cover", type=int, default=5, help="(-cover, cover)")
    parser.add_argument("--geogi_epsilon", type=float, default=8, help="epsilon in geogi")

    args = parser.parse_args()

    main(args)
