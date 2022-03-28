from .mpii import Mpii



def get_mpii_data(args):
    dataset = {}
    dataset['train'] = Mpii(is_train=True,
                            image_path=args.dataset_path+'/images/',
                            anno_path=args.dataset_path+'/mpii_annotations.json',
                            inp_res=256, out_res=64, sigma=1, scale_factor=0.25,
                            rot_factor=30, label_type='Gaussian')
    dataset['unlabeled'] = Mpii(is_train=True,
                                image_path=args.dataset_path + '/images/',
                                anno_path=args.dataset_path + '/mpii_annotations.json',
                                inp_res=256, out_res=64, sigma=1, scale_factor=0.25,
                                rot_factor=30, label_type='Gaussian')
    dataset['test'] = Mpii(is_train=False,
                           image_path=args.dataset_path + '/images/',
                           anno_path=args.dataset_path + '/mpii_annotations.json',
                           inp_res=256, out_res=64, sigma=1, scale_factor=0.25,
                           rot_factor=30, label_type='Gaussian')
    return dataset