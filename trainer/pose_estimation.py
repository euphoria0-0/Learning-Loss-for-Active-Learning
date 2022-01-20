from .HPEloss import JointsMSELoss
from utils.utils import AverageMeter
from utils.hpe_eval import *
from torch import optim
import torch
import time
from tqdm import tqdm
import matplotlib.pyplot as plt


idx = [1,2,3,4,5,6,11,12,15,16]

class HPETrainer:
    def __init__(self, model, dataloaders, writer, args):
        self.device = args.device
        self.model = model
        self.dataloaders = dataloaders
        self.batch_size = args.batch_size
        self.num_epoch = args.num_epoch
        self.args = args
        self.writer = writer
        self.lr = args.lr
        self.njoints = args.nJoint

        # loss function
        self.criterion = JointsMSELoss().to(args.device)

        self.optimizer = {
            'backbone': optim.RMSprop(self.model['backbone'].parameters(), lr=args.lr, weight_decay=args.wdecay),
            'module': optim.RMSprop(self.model['module'].parameters(), lr=args.lr, weight_decay=args.wdecay)
        }

        self.lr_scheduler = {
            'backbone': optim.lr_scheduler.MultiStepLR(self.optimizer['backbone'], milestones=args.milestone, gamma=0.1),
            'module': optim.lr_scheduler.MultiStepLR(self.optimizer['module'], milestones=args.milestone, gamma=0.1)
        }

    def train(self, debug=False, flip=True):
        # train and eval
        print('>> Train')
        for epoch in range(self.num_epoch):
            # decay sigma
            if self.args.sigma_decay > 0:
                self.dataloaders['train'].dataset.sigma *= self.args.sigma_decay

            # train for one epoch
            train_loss, train_acc = self.train_epoch(debug, flip)
            print('Epoch {}/{} TrainLoss {:.6f} TrainAcc {:.4f}'.format(epoch + 1, self.num_epoch, train_loss, train_acc))

            self.lr_scheduler['backbone'].step()
            self.lr_scheduler['module'].step()

            # append logger file
            self.writer.add_scalars('Train', {
                'Loss': train_loss,
                'Acc': train_acc
            }, epoch)


    def train_epoch(self, debug=False, flip=True):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        acces = AverageMeter()

        # switch to train mode
        self.model['backbone'].train()
        self.model['module'].train()

        end = time.time()

        gt_win, pred_win = None, None
        #bar = Bar('Train', max=len(train_loader))
        for i, (input, target, meta) in enumerate(self.dataloaders['train']):
            # measure data loading time
            data_time.update(time.time() - end)

            input, target = input.to(self.device), target.to(self.device, non_blocking=True)
            target_weight = meta['target_weight'].to(self.device, non_blocking=True)

            # compute output
            output = self.model['backbone'](input)
            if type(output) == list:  # multiple output
                loss = 0
                for o in output:
                    loss += self.criterion(o, target, target_weight)
                output = output[-1]
            else:  # single output
                loss = self.criterion(output, target, target_weight)

            # acc
            acc = accuracy(output, target, idx)

            # visualize groundtruth and predictions
            if debug:
                gt_batch_img = batch_with_heatmap(input, target)
                pred_batch_img = batch_with_heatmap(input, output)
                if not gt_win or not pred_win:
                    ax1 = plt.subplot(121)
                    ax1.title.set_text('Groundtruth')
                    gt_win = plt.imshow(gt_batch_img)
                    ax2 = plt.subplot(122)
                    ax2.title.set_text('Prediction')
                    pred_win = plt.imshow(pred_batch_img)
                else:
                    gt_win.set_data(gt_batch_img)
                    pred_win.set_data(pred_batch_img)
                plt.pause(.05)
                plt.draw()

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            acces.update(acc[0], input.size(0))

            # compute gradient and do SGD step
            self.optimizer['backbone'].zero_grad()
            self.optimizer['module'].zero_grad()
            loss.backward()
            self.optimizer['backbone'].step()
            self.optimizer['module'].step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


        return losses.avg, acces.avg

    def test(self, model, round=None, phase='Test', debug=False, flip=True):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        acces = AverageMeter()

        # predictions
        val_loader = self.dataloaders[phase.lower()]
        predictions = torch.Tensor(val_loader.dataset.__len__(), self.njoints, 2)

        # switch to evaluate mode
        model['backbone'].eval()
        model['module'].eval()

        gt_win, pred_win = None, None
        end = time.time()

        with torch.no_grad():
            for i, (input, target, meta) in tqdm(enumerate(val_loader)):
                # measure data loading time
                data_time.update(time.time() - end)

                input = input.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                target_weight = meta['target_weight'].to(self.device, non_blocking=True)

                # compute output
                output = model['backbone'](input)
                score_map = output[-1].cpu() if type(output) == list else output.cpu()
                if flip:
                    flip_input = torch.from_numpy(fliplr(input.clone().cpu().numpy())).float().to(self.device)
                    flip_output = model['backbone'](flip_input)
                    flip_output = flip_output[-1].cpu() if type(flip_output) == list else flip_output.cpu()
                    flip_output = flip_back(flip_output)
                    score_map += flip_output

                if type(output) == list:  # multiple output
                    loss = 0
                    for o in output:
                        loss += self.criterion(o, target, target_weight)
                    output = output[-1]
                else:  # single output
                    loss = self.criterion(output, target, target_weight)

                acc = accuracy(score_map, target.cpu(), idx)

                # generate predictions
                preds = final_preds(score_map, meta['center'], meta['scale'], [64, 64])
                for n in range(score_map.size(0)):
                    predictions[meta['index'][n], :, :] = preds[n, :, :]

                if debug:
                    gt_batch_img = batch_with_heatmap(input, target)
                    pred_batch_img = batch_with_heatmap(input, score_map)
                    if not gt_win or not pred_win:
                        plt.subplot(121)
                        gt_win = plt.imshow(gt_batch_img)
                        plt.subplot(122)
                        pred_win = plt.imshow(pred_batch_img)
                    else:
                        gt_win.set_data(gt_batch_img)
                        pred_win.set_data(pred_batch_img)
                    plt.pause(.05)
                    plt.draw()

                # measure accuracy and record loss
                losses.update(loss.item(), input.size(0))
                acces.update(acc[0], input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()


        self.writer.add_scalars('Test', {
            'Acc': acces.avg
        }, round)

        return losses.avg, acces.avg, predictions
