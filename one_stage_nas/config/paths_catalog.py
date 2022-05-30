import os


class DatasetCatalog(object):
    DATA_DIR = '/home/path/datasets'
    DATASETS = {
        "cityscapes_fine_train": {
            "data_file": "cityscapes/lists/train.lst",
            "data_dir": "cityscapes"
        },
        "cityscapes_fine_val": {
            "data_file": "cityscapes/lists/val.lst",
            "data_dir": "cityscapes"
        }
    }

    @staticmethod
    def get(name):
        if 'cityscapes' in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_file=os.path.join(data_dir, attrs['data_file']),
                data_dir=os.path.join(data_dir, attrs['data_dir']))
            return dict(
                factory="CityscapesDataset",
                args=args)
        raise RuntimeError("Dataset not available: {}".format(name))



