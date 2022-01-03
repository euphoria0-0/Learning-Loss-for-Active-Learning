'''
Learning Loss for Active Learning

'''
import argparse
from tensorboardX import SummaryWriter

from data.voc_data import *
from active_learner import *
from model import *
from trainer import Trainer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0', help='gpu cuda index')

    parser.add_argument('--dataset', help='dataset', type=str, default='VOC0712')
    parser.add_argument('--dataset_path', help='data path', type=str, default='D:/data/detection/VOCdevkit')
    parser.add_argument('--save_path', help='save path', type=str, default='./results/')

    parser.add_argument('--num_trial', type=int, default=5, help='number of trials')
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

    writer = SummaryWriter()

    # load data
    dataset = get_data(args)
    args.nTrain = len(dataset['train'])
    args.nClass = 21

    # active learning round
    for round in range(len(args.query_size)):
        nLabeled = args.query_size[round]
        nQuery = args.query_size[round + 1] - args.query_size[round] if round < len(args.query_size) - 1 else 'X'
        print(f'> round {round + 1}/{len(args.query_size)} Labeled {nLabeled} Query {nQuery}')

        # set model
        model = get_model(args)
        model['backbone'] = model['backbone'].to(args.device)
        model['module'] = model['module'].to(args.device)
        if ',' in args.gpu_id:
            model['backbone'] = torch.nn.DataParallel(model['backbone'])
            model['module'] = torch.nn.DataParallel(model['module'])
            torch.backends.cudnn.benchmark = True

        # set active learner
        active_learner = LearningLoss(dataset, args)
        dataloaders = active_learner.get_current_dataloaders()

        # train
        trainer = Trainer(model, dataloaders, writer, args)
        trainer.train()

        # test / inference
        model = get_model(args, phase='test')
        model['backbone'] = model['backbone'].to(args.device)
        model['module'] = model['module'].to(args.device)
        trainer.test(model, round=round, phase='test')

        # query
        query_model = {'backbone': model['backbone'], 'module': trainer.model['module']}
        if round < len(args.query_size) - 1:
            active_learner.query(nQuery, query_model)
