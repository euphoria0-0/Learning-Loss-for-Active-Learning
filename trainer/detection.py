import torch
from torch import optim
from torch import nn
from tqdm import tqdm

from model.ssd_pytorch.ssd_layers import MultiBoxLoss
from model.lossnet import LossPredLoss
from utils import *


class DetectionTrainer:
    def __init__(self, model, dataloaders, writer, args):
        self.device = args.device
        self.model = model
        self.dataloaders = dataloaders
        self.test_dataset = dataloaders['test'].dataset
        self.batch_size = args.batch_size
        #self.num_epoch = args.num_epoch
        self.args = args
        self.writer = writer

        # loss function
        self.criterion = MultiBoxLoss(args.nClass, 0.5, True, 0, True, 3, 0.5, False, args.device)

        self.optimizer = {
            'backbone': optim.SGD(self.model['backbone'].parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wdecay),
            'module': optim.SGD(self.model['module'].parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wdecay)
        }

        # self.lr_scheduler = {
        #     'backbone': optim.lr_scheduler.MultiStepLR(self.optimizer['backbone'], gamma=args.gamma, milestones=args.milestone),
        #     'module': optim.lr_scheduler.MultiStepLR(self.optimizer['module'], gamma=args.gamma, milestones=args.milestone)
        # }

    def train(self):
        self.model['backbone'].train()
        self.model['module'].train()

        # loss counters
        loc_loss, conf_loss = 0, 0
        epoch = 0
        step_index = 0
        epoch_size = self.args.nTrain // self.batch_size

        # create batch iterator
        # batch_iterator = iter(data_loader)
        batch_iterator = iter(cycle(self.dataloaders['train']))

        for iteration in tqdm(range(self.args.start_iter, self.args.max_iter)):
            if iteration != 0 and (iteration % epoch_size == 0):
                # reset epoch loss counters
                loc_loss, conf_loss = 0, 0
                epoch += 1

            if iteration in self.args.lr_steps:
                step_index += 1
                self.adjust_learning_rate(step_index)

            # load train data
            images, targets = next(batch_iterator)
            images = torch.autograd.Variable(images.to(self.device))
            targets = [torch.autograd.Variable(ann.to(self.device), volatile=True) for ann in targets]

            # forward
            t0 = time.time()
            out = self.model['backbone'](images)
            features = self.model['backbone'].get_features()

            # backprop
            self.optimizer['backbone'].zero_grad()
            loss_l, loss_c = self.criterion(out, targets)
            target_loss = loss_l + loss_c

            if epoch > self.args.epoch_loss:
                # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model.
                for i in range(len(features)):
                    features[i] = features[i].detach()

            pred_loss = self.model['module'](features)
            #pred_loss = pred_loss.view(pred_loss.size(0))

            #backbone_loss = torch.sum(target_loss) / target_loss.size(0)
            backbone_loss = target_loss
            module_loss = LossPredLoss(pred_loss, target_loss.reshape(1), margin=1.0)
            loss = backbone_loss + self.args.weights * module_loss

            loss.backward()
            self.optimizer['backbone'].step()

            t1 = time.time()
            loc_loss += loss_l.data
            conf_loss += loss_c.data

            self.writer.add_scalars('Loss', {
                'train': loss,
            }, epoch)
            # self.writer.add_scalars('Acc', {
            #     'train': train_acc,
            #     'valid': val_acc
            # }, epoch)

            if iteration % 10000 == 0:
                print('timer: %.4f sec.' % (t1 - t0))
                #print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data), end=' ')
                print('Epoch {} Iter {}/{} TrainLoss {:.6f}'.format(epoch, iteration, self.args.max_iter, loss), end=' ')

                self.writer.flush()

            if iteration != 0 and iteration % 5000 == 0:
                torch.save(self.model['backbone'].state_dict(), f'weights/ssd300_VOC_{iteration}.pth')

            torch.cuda.empty_cache()

        torch.save(self.model['backbone'].state_dict(), f'{self.args.save_path}VOC.pth')

        #return train_acc

    def test(self, model, round=0, phase='test'):
        model['backbone'].eval()
        model['module'].eval()

        # evaluation
        num_images = len(self.test_dataset)
        # all detections are collected into:
        #    all_boxes[cls][image] = N x 5 array of detections in
        #    (x1, y1, x2, y2, score)
        all_boxes = [[[] for _ in range(num_images)]
                     for _ in range(self.args.nClass)]

        # timers
        _t = {'im_detect': Timer(), 'misc': Timer()}
        output_dir = get_output_dir('ssd300_120000', phase)
        det_file = os.path.join(output_dir, 'detections.pkl')

        for i in tqdm(range(num_images), desc='> im_detect'):
            im, gt, h, w = self.test_dataset.pull_item(i)

            x = torch.autograd.Variable(im.unsqueeze(0))
            x = x.to(self.device)
            _t['im_detect'].tic()
            detections = model['backbone'](x).data
            detect_time = _t['im_detect'].toc(average=False)

            # skip j = 0, because it's the background class
            for j in range(1, detections.size(1)):
                dets = detections[0, j, :]
                mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
                dets = torch.masked_select(dets, mask).view(-1, 5)
                if dets.size(0) == 0:
                    continue
                boxes = dets[:, 1:]
                boxes[:, 0] *= w
                boxes[:, 2] *= w
                boxes[:, 1] *= h
                boxes[:, 3] *= h
                scores = dets[:, 0].cpu().numpy()
                cls_dets = np.hstack((boxes.cpu().numpy(),
                                      scores[:, np.newaxis])).astype(np.float32,
                                                                     copy=False)
                all_boxes[j][i] = cls_dets

            #print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1, num_images, detect_time))

        with open(det_file, 'wb') as f:
            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

        print('Evaluating detections')
        #evaluate_detections(all_boxes, output_dir, dataset)
        write_voc_results_file(all_boxes, self.test_dataset, self.args.dataset_path)
        mAP = do_python_eval(self.args.dataset_path, output_dir)

        print(' Test mAP: {:.4f}'.format(mAP))

        self.writer.add_scalars('Test_mAP', {
            'test': mAP
        }, round)
        self.writer.flush()

        torch.cuda.empty_cache()
        return mAP

    def get_model(self):
        return self.model

    def adjust_learning_rate(self, step):
        """Sets the learning rate to the initial LR decayed by 10 at every
            specified step
        # Adapted from PyTorch Imagenet example:
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py
        """
        # lr = args.lr * (gamma ** (step))
        lr = self.args.lr * (self.args.gamma ** (step))
        for param_group in self.optimizer['backbone'].param_groups:
            param_group['lr'] = lr

