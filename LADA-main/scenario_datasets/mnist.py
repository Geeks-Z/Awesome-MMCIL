import os
from torchvision import datasets, transforms
from .utils import DatasetBase, Datum


class MNIST(DatasetBase):

    dataset_dir = 'MNIST'

    def __init__(self, root, num_shots, preprocess=None, val_transform=None):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.preprocess = preprocess or transforms.ToTensor()
        self.val_transform = val_transform or transforms.ToTensor()

        train_full = datasets.MNIST(
            self.dataset_dir, train=True, download=True, transform=None
        )
        test_full = datasets.MNIST(
            self.dataset_dir, train=False, download=True, transform=None
        )
        
        train_data = [
            Datum(impath=img, label=label, classname=str(label))
            for img, label in train_full
        ]

        test_data = [
            Datum(impath=img, label=label, classname=str(label))
            for img, label in test_full
        ]

        train = self.generate_fewshot_dataset(train_data, num_shots=num_shots)

        super().__init__(train_x=train, val=None, test=test_data)
        classnames = train_full.classes
        classnames = ['number: "{}"'.format(classname) for classname in classnames] # To keep consistent with prompt
        self.update_classnames(classnames)