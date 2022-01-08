'''
Learning Loss for Active Learning
'''
import os
import argparse

from tensorboardX import SummaryWriter

from data import voc, cifar
from active_learner import *
from trainer import *
from model import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='detection', choices=['clf','detection'])

    parser.add_argument('--gpu_id', type=str, default='0', help='gpu cuda index')

    parser.add_argument('--dataset', help='dataset', type=str, default='VOC0712')
    parser.add_argument('--dataset_path', help='data path', type=str, default='D:/data/detection/VOCdevkit')
    parser.add_argument('--save_path', help='save path', type=str, default='./weights/')

    parser.add_argument('--num_trial', type=int, default=1, help='number of trials')
    parser.add_argument('--num_epoch', type=int, default=300, help='number of epochs')

    parser.add_argument('--batch_size', type=int, default=32, help='Batch size used for training only')
    parser.add_argument('--query_size', help='number of points at each round', type=list,
                        default=[1000,2000,3000,4000,5000,6000,7000,8000,9000,10000])

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
    parser.add_argument('--wdecay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--milestone', type=list, default=[160], help='number of acquisition')

    parser.add_argument('--epoch_loss', type=int, default=240,
                        help='After 120 epochs, stop the gradient from the loss prediction module propagated to the target model')
    parser.add_argument('--margin', type=float, default=1.0, help='MARGIN')
    parser.add_argument('--weights', type=float, default=1.0, help='weight')
    parser.add_argument('--subset', type=int, default=10000, help='subset for learning loss')

    parser.add_argument('--start_iter', default=0, type=int, help='Resume training at this iter')
    parser.add_argument('--max_iter', type=int, default=120000, help='')
    parser.add_argument('--lr_steps', type=list, default=[80000, 100000, 120000], help='')

    parser.add_argument('--fix_seed', action='store_true', default=False, help='fix seed for reproducible')
    parser.add_argument('--seed', type=int, default=0, help='seed number')

    parser.add_argument('--resume', default=None, type=str,
                        help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--confidence_threshold', default=0.01, type=float,
                        help='Detection confidence threshold')
    parser.add_argument('--top_k', default=5, type=int,
                        help='Further restrict the number of predictions to parse')

    args = parser.parse_args()
    return args


def get_dataset(args):
    if args.task == 'clf':
        dataset = cifar.CIFARDataset(args)
        args.nTrain, args.nClass = dataset.nTrain, dataset.nClass
        dataset = dataset.dataset
    elif args.task == 'detection':
        dataset = voc.get_voc_data(args)
        args.nTrain, args.nClass = len(dataset['train']), 21
    return dataset, args

def get_model(args):
    if args.task == 'clf':
        model = get_resnet_model(args)
    elif args.task == 'detection':
        model = get_ssd_model(args)
    return model

def get_inference_model(args, trained_model):
    if args.task == 'clf':
        model = trained_model
    elif args.task == 'detection':
        test_model = get_ssd_model(args, phase='test')
        model = {'backbone': test_model['backbone'], 'module': trained_model['module']}
    return model

def get_trainer(args, model, dataloaders, writer):
    if args.task == 'clf':
        trainer = ClassificationTrainer(model, dataloaders, writer, args)
    elif args.task == 'detection':
        trainer = DetectionTrainer(model, dataloaders, writer, args)
    return trainer


if __name__ == '__main__':
    args = get_args()
    print('=' * 100)
    print('Arguments = ')
    for arg in vars(args):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 100)

    args.device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.device)  # change allocation of current GPU
    print(f'Current cuda device: {torch.cuda.current_device()}')

    torch.set_default_tensor_type('torch.FloatTensor')

    os.makedirs(args.save_path, exist_ok=True)
    writer = SummaryWriter()

    # load data
    dataset, args = get_dataset(args)

    # trial
    for trial in range(args.num_trial):
        print(f'>> TRIAL {trial + 1}/{args.num_trial}')

        # set active learner
        active_learner = LearningLoss(dataset, args)

        # active learning round
        for round in range(len(args.query_size)):
            nLabeled = args.query_size[round]
            nQuery = args.query_size[round + 1] - args.query_size[round] if round < len(args.query_size) - 1 else 'X'
            print(f'> ROUND {round + 1}/{len(args.query_size)} Labeled {nLabeled} Query {nQuery}')

            # set model
            model = get_model(args)

            # get current data
            dataloaders = active_learner.get_current_dataloaders()

            # train
            trainer = get_trainer(args, model, dataloaders, writer)
            trainer.train()

            # test / inference
            inference_model = get_inference_model(args, trainer.model)
            trainer.test(inference_model, round=round, phase='test')

            # query
            if round < len(args.query_size) - 1:
                active_learner.query(nQuery, inference_model)
