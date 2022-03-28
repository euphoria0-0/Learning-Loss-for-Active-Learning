import numpy as np
import torchvision.transforms as T
import torchvision.datasets as D


class CIFARDataset:
    def __init__(self, args):
        self.data = args.dataset
        self.data_dir = args.dataset_path
        self.al_method = 'learningloss' #args.al_method

        if self.data == 'CIFAR10':
            self.nClass = 10
            self.nTrain = 50000
            self.nTest = 10000

        elif self.data == 'CIFAR100':
            self.nClass = 100
            self.nTrain = 50000
            self.nTest = 10000

        self.dataset = {}
        self._getData()


    def _getData(self):
        self._data_transform()
        if self.data == 'CIFAR10':
            self.dataset['train'] = D.CIFAR10(self.data_dir+'cifar10', train=True, download=True, transform=self.train_transform)
            self.dataset['unlabeled'] = D.CIFAR10(self.data_dir+'cifar10', train=True, download=True, transform=self.test_transform)
            self.dataset['test'] = D.CIFAR10(self.data_dir+'cifar10', train=False, download=True, transform=self.test_transform)
            self.dataset['label'] = self.dataset['train'].targets

        elif self.data == 'CIFAR100':
            self.dataset['train'] = D.CIFAR100(self.data_dir+'cifar100', train=True, download=True, transform=self.train_transform)
            self.dataset['unlabeled'] = D.CIFAR100(self.data_dir+'cifar100', train=True, download=True, transform=self.test_transform)
            self.dataset['test'] = D.CIFAR100(self.data_dir+'cifar100', train=False, download=True, transform=self.test_transform)
            self.dataset['label'] = self.dataset['train'].targets

        # self.dataset['train'] = Dataset_idx(self.dataset['train'])
        # self.dataset['unlabeled'] = Dataset_idx(self.dataset['unlabeled'])
        # self.dataset['test'] = Dataset_idx(self.dataset['test'])


    def _data_transform(self):
        if self.data == 'CIFAR10':
            mean, std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
            if self.al_method == 'learningloss':
                add_transform = [T.RandomHorizontalFlip(),
                                 T.RandomCrop(size=32, padding=4)]
            else:
                add_transform = []

        elif self.data == 'CIFAR100':
            mean, std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
            if self.al_method == 'learningloss':
                add_transform = [T.RandomHorizontalFlip(),
                                 T.RandomCrop(size=32, padding=4)]
            else:
                add_transform = []

        base_transform = [T.Resize((224,224)), T.ToTensor(), T.Normalize(mean, std)]
        self.test_transform = T.Compose(base_transform)
        self.train_transform = T.Compose(add_transform + base_transform)



class Dataset_idx:
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        if isinstance(index, np.float64):
            index = index.astype(np.int64)
        data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
       return len(self.dataset)


