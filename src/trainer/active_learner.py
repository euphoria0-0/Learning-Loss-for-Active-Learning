import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from ..data.voc import detection_collate


class ActiveLearner:
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.init_size = args.query_size[0]
        self.device = args.device
        self.nTrain = args.nTrain
        self.args = args
        self.loader_args = {'batch_size': args.batch_size, 'pin_memory': True, 'shuffle': False}
        if args.task == 'detection':
            self.loader_args['collate_fn'] = detection_collate
        self._init_setting()

    def _init_setting(self):
        total_indices = np.arange(self.nTrain)
        print(f'nTrain: {self.nTrain} nInit: {self.init_size}')
        self.labeled_indices = np.random.choice(total_indices, self.init_size, replace=False).tolist()
        self.unlabeled_indices = list(set(total_indices) - set(self.labeled_indices))
        self.dataloaders = self._get_dataloaders()

    def _get_dataloaders(self):
        dataloaders = {
            'train': DataLoader(self.dataset['train'], **self.loader_args,
                                sampler=SubsetRandomSampler(self.labeled_indices)),
            'unlabeled': DataLoader(self.dataset['unlabeled'], **self.loader_args,
                                    sampler=SubsetRandomSampler(self.unlabeled_indices)),
            'test': DataLoader(self.dataset['test'], **self.loader_args)
        }
        return dataloaders

    def update(self, query_indices):
        print(f'selected data: {sorted(query_indices)[:10]}')
        self.labeled_indices += query_indices
        self.unlabeled_indices = list(set(self.unlabeled_indices) - set(query_indices))
        self.dataloaders = self._get_dataloaders()

    def get_current_dataloaders(self):
        return self.dataloaders

    def query(self, n, model):
        pass



class RandomSampling(ActiveLearner):
    def __init__(self, dataset, args):
        super().__init__(dataset, args)

    def query(self, nQuery, model=None):
        query_indices = np.random.choice(self.unlabeled_indices, nQuery, replace=False).tolist()
        self.labeled_indices += query_indices
        self.unlabeled_indices = list(set(self.unlabeled_indices) - set(query_indices))



class LearningLoss(ActiveLearner):
    def __init__(self, dataset, args):
        super().__init__(dataset, args)
        self.subset = args.subset if args.task != 'detection' else None


    def query(self, nQuery, model):
        if self.subset is not None:
            subset = np.random.choice(self.unlabeled_indices, self.subset, replace=False)
        else:
            subset = np.array(self.unlabeled_indices)
        unlabeled_loader = DataLoader(self.dataset['unlabeled'], **self.loader_args,
                                      sampler=SubsetRandomSampler(subset))
        model['backbone'].eval()
        model['module'].eval()

        uncertainty = torch.tensor([])
        with torch.no_grad():
            for data in tqdm(unlabeled_loader, desc='> inference of unlabeled data'):
                inputs = data[0].to(self.device)

                _ = model['backbone'](inputs)
                features = model['backbone'].get_features()
                pred_loss = model['module'](features)
                pred_loss = pred_loss.view(pred_loss.size(0))
                uncertainty = torch.cat((uncertainty, pred_loss.cpu().data), 0)

                torch.cuda.empty_cache()

        arg = np.argsort(uncertainty.numpy())[-nQuery:]
        query_indices = subset[arg].tolist()
        self.update(query_indices)