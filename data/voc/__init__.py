'''
Reference:
    https://github.com/amdegroot/ssd.pytorch
'''
from .augmentations import SSDAugmentation
from .voc0712 import *


voc_means = (104, 117, 123)

voc_classes = VOC_CLASSES #(  # always index 0
    # 'aeroplane', 'bicycle', 'bird', 'boat',
    # 'bottle', 'bus', 'car', 'cat', 'chair',
    # 'cow', 'diningtable', 'dog', 'horse',
    # 'motorbike', 'person', 'pottedplant',
    # 'sheep', 'sofa', 'train', 'tvmonitor')

# SSD300 CONFIGS
voc_cfg = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}


def get_voc_data(args):
    dataset = {}
    dataset['train'] = VOCDetection(root=args.dataset_path,
                                    transform=SSDAugmentation(voc_cfg['min_dim'], voc_means))
    dataset['unlabeled'] = VOCDetection(root=args.dataset_path,
                                        transform=BaseTransform(300, voc_means),
                                        target_transform=VOCAnnotationTransform())
    dataset['test'] = VOCDetection(root=args.dataset_path,
                                   image_sets=[('2007', 'test')],
                                   transform=BaseTransform(300, voc_means),
                                   target_transform=VOCAnnotationTransform())
    return dataset



def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


def base_transform(image, size, mean):
    x = cv2.resize(image, (size, size)).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x


class BaseTransform:
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size, self.mean), boxes, labels
