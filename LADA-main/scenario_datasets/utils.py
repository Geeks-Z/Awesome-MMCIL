import os
import random
import os.path as osp
import tarfile
import zipfile
from collections import defaultdict
import gdown
import json

from torch.utils.data import Dataset
from PIL import Image


def split_trainval(trainval, p_val=0.2):
    p_trn = 1 - p_val
    print(f'Splitting trainval into {p_trn:.0%} train and {p_val:.0%} val')
    tracker = defaultdict(list)
    for idx, item in enumerate(trainval):
        label = item.label
        tracker[label].append(idx)

    train, val = [], []
    for label, idxs in tracker.items():
        n_val = round(len(idxs) * p_val)
        assert n_val > 0
        random.shuffle(idxs)
        for n, idx in enumerate(idxs):
            item = trainval[idx]
            if n < n_val:
                val.append(item)
            else:
                train.append(item)

    return train, val


def save_split(train, val, test, filepath, path_prefix):
    def _extract(items):
        out = []
        for item in items:
            impath = item.impath
            label = item.label
            classname = item.classname
            impath = impath.replace(path_prefix, '')
            if impath.startswith('/'):
                impath = impath[1:]
            out.append((impath, label, classname))
        return out

    train = _extract(train)
    val = _extract(val)
    test = _extract(test)

    split = {
        'train': train,
        'val': val,
        'test': test
    }

    write_json(split, filepath)
    print(f'Saved split to {filepath}')


def read_split(filepath, path_prefix):
    def _convert(items):
        out = []
        for impath, label, classname in items:
            impath = os.path.join(path_prefix, impath)
            item = Datum(
                impath=impath,
                label=int(label),
                classname=classname
            )
            out.append(item)
        return out

    print(f'Reading split from {filepath}')
    split = read_json(filepath)
    train = _convert(split['train'])
    val = _convert(split['val'])
    test = _convert(split['test'])

    return train, val, test


def read_json(fpath):
    """Read json file from a path."""
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    """Writes to a json file."""
    if not osp.exists(osp.dirname(fpath)):
        os.makedirs(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def read_image(path):
    """Read image from path using ``PIL.Image``.

    Args:
        path (str): path to an image.

    Returns:
        PIL image
    """
    if not osp.exists(path):
        raise IOError('No file exists at {}'.format(path))

    while True:
        try:
            img = Image.open(path).convert('RGB')
            return img
        except IOError:
            print(
                'Cannot read image from {}, '
                'probably due to heavy IO. Will re-try'.format(path)
            )


def listdir_nohidden(path, sort=False):
    """List non-hidden items in a directory.

    Args:
         path (str): directory path.
         sort (bool): sort the items.
    """
    items = [f for f in os.listdir(path) if not f.startswith('.') and 'sh' not in f]
    if sort:
        items.sort()
    return items


class Datum:
    """Data instance which defines the basic attributes.

    Args:
        impath (str): image path.
        label (int): class label.
        domain (int): domain label.
        classname (str): class name.
    """

    def __init__(self, impath='', label=0, domain=-1, classname=''):
        # assert isinstance(impath, str)
        assert isinstance(label, int)
        assert isinstance(domain, int)
        assert isinstance(classname, str)

        self._impath = impath
        self._label = label
        self._domain = domain
        self._classname = classname

    @property
    def impath(self):
        return self._impath

    @property
    def label(self):
        return self._label

    def update_label(self, base_cls_num):
        self._label += base_cls_num

    @property
    def domain(self):
        return self._domain

    @property
    def classname(self):
        return self._classname

    def update_classname(self, new_classname):
        self._classname = new_classname


class DatasetBase:
    """A unified dataset class for
    1) domain adaptation
    2) domain generalization
    3) semi-supervised learning
    """
    dataset_dir = ''  # the directory where the dataset is stored
    domains = []  # string names of all domains

    def __init__(self, train_x=None, train_u=None, val=None, test=None):
        self._train_x = train_x  # labeled training data
        self._train_u = train_u  # unlabeled training data (optional)
        self._val = val  # validation data (optional)
        self._test = test  # test data

        self._num_classes = self.get_num_classes(train_x)
        self._lab2cname, self._classnames = self.get_lab2cname(train_x)

    @property
    def train_x(self):
        return self._train_x

    @property
    def train_u(self):
        return self._train_u

    @property
    def val(self):
        return self._val

    @property
    def test(self):
        return self._test

    @property
    def lab2cname(self):
        return self._lab2cname

    @property
    def classnames(self):
        return self._classnames

    def update_classnames(self, new_classnames):
        self._classnames = new_classnames

    @property
    def num_classes(self):
        return self._num_classes

    def merge_dataset(self, data_source):
        self._train_x += data_source.train_x
        self._val += data_source.val
        self._test += data_source.test

    def get_num_classes(self, data_source):
        """Count number of classes.

        Args:
            data_source (list): a list of Datum objects.
        """
        if len(data_source) == 0:
            return 0
        label_set = set()
        for item in data_source:
            label_set.add(item.label)
        return max(label_set) + 1

    def update_num_classes(self, data_source):
        self._num_classes = self.get_num_classes(data_source)
        self._lab2cname, self._classnames = self.get_lab2cname(data_source)

    def get_lab2cname(self, data_source):
        """Get a label-to-classname mapping (dict).

        Args:
            data_source (list): a list of Datum objects.
        """
        if len(data_source) == 0:
            return None, None
        container = set()
        for item in data_source:
            container.add((item.label, item.classname))
        mapping = {label: classname for label, classname in container}
        labels = list(mapping.keys())
        labels.sort()
        classnames = [mapping[label] for label in labels]
        return mapping, classnames

    def check_input_domains(self, source_domains, target_domains):
        self.is_input_domain_valid(source_domains)
        self.is_input_domain_valid(target_domains)

    def is_input_domain_valid(self, input_domains):
        for domain in input_domains:
            if domain not in self.domains:
                raise ValueError(
                    'Input domain must belong to {}, '
                    'but got [{}]'.format(self.domains, domain)
                )

    def download_data(self, url, dst, from_gdrive=True):
        if not osp.exists(osp.dirname(dst)):
            os.makedirs(osp.dirname(dst))

        if from_gdrive:
            gdown.download(url, dst, quiet=False)
        else:
            raise NotImplementedError

        print('Extracting file ...')

        try:
            tar = tarfile.open(dst)
            tar.extractall(path=osp.dirname(dst))
            tar.close()
        except:
            zip_ref = zipfile.ZipFile(dst, 'r')
            zip_ref.extractall(osp.dirname(dst))
            zip_ref.close()

        print('File extracted to {}'.format(osp.dirname(dst)))

    def generate_fewshot_dataset(
            self, *data_sources, num_shots=-1, repeat=True
    ):
        """Generate a few-shot dataset (typically for the training set).

        This function is useful when one wants to evaluate a model
        in a few-shot learning setting where each class only contains
        a few number of images.

        Args:
            data_sources: each individual is a list containing Datum objects.
            num_shots (int): number of instances per class to sample.
            repeat (bool): repeat images if needed.
        Return:
            data_sources: each individual is a list containing Datum objects. (Sampled)
        """
        if num_shots < 1:
            if len(data_sources) == 1:
                return data_sources[0]
            return data_sources

        print(f'Creating a {num_shots}-shot dataset')

        output = []

        for data_source in data_sources:
            tracker = self.split_dataset_by_label(data_source)
            dataset = []

            for label, items in tracker.items():
                if len(items) >= num_shots:
                    sampled_items = random.sample(items, num_shots)
                else:
                    if repeat:
                        sampled_items = random.choices(items, k=num_shots)
                    else:
                        sampled_items = items
                dataset.extend(sampled_items)

            output.append(dataset)

        if len(output) == 1:
            return output[0]

        return output

    def split_dataset_by_label(self, data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into class-specific groups stored in a dictionary.

        Args:
            data_source (list): a list of Datum objects.
        """
        output = defaultdict(list)

        for item in data_source:
            output[item.label].append(item)

        return output

    def split_dataset_by_domain(self, data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into domain-specific groups stored in a dictionary.

        Args:
            data_source (list): a list of Datum objects.
        """
        output = defaultdict(list)

        for item in data_source:
            output[item.domain].append(item)

        return output


class DatasetWrapper(Dataset):
    def __init__(self, data_source, transform=None):
        self.data_source = data_source
        self.transform = transform

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        if isinstance(item.impath, str):
            img0 = Image.open(item.impath).convert("RGB")
        else:
            img0 = item.impath.convert("RGB")

        if self.transform is not None:
            img = self.transform(img0)

        return img, item.label


class CustomConcatDataset(Dataset):
    def __init__(self, dataset_instances, class_offset):
        self.dataset_instances = dataset_instances
        self.class_offset = class_offset

    def __len__(self):
        return sum(len(ds) for ds in self.dataset_instances)

    def __getitem__(self, index):
        for i, dataset in enumerate(self.dataset_instances):
            if index < len(dataset):
                data, target = dataset[index]
                target += self.class_offset[i]  # Adjust the target based on the class offset
                return data, target
            index -= len(dataset)

        raise IndexError("Index out of range.")
