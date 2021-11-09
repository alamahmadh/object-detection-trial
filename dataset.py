import os
import torch
from skimage.io import imread
from torchvision.ops import box_convert
from typing import Dict


class ODDataSet(torch.utils.data.Dataset):
    """
    Builds a dataset with images and their respective targets.
    A target is expected to be a pickled file of a dict
    and should contain at least a 'boxes' and a 'labels' key.
    In case your labels are strings, you can use mapping (a dict) to int-encode them.
    Returns a dict with the following keys: 'x', 'x_name', 'y', 'y_name'
    """

    def __init__(self,
                 df: pd.DataFrame = None,
                 img_dir: str = '',
                 transform: ComposeDouble = None,
                 use_cache: bool = False,
                 convert_to_format: str = None,
                 mapping: Dict = None
                 ):
        self.df = df
        self.img_dir = img_dir
        self.inputs = self.df['filename'].unique()
        self.transform = transform
        self.use_cache = use_cache
        self.convert_to_format = convert_to_format
        self.mapping = mapping

        if self.use_cache:
            # Use multiprocessing to load images and targets into RAM
            from multiprocessing import Pool
            with Pool() as pool:
                self.cached_data = pool.starmap(self.read_images, os.path.join(self.img_dir, self.inputs))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,
                    index: int):
        if self.use_cache:
            x, y = self.cached_data[index]
        else:
            # Load input image and target (boxes and labels)
            x = self.read_images(os.path.join(self.img_dir, self.inputs[index]))
            image_data = self.df.loc[self.df['filename'] == self.inputs[index]]
            y = image_data[['xmin', 'ymin', 'xmax', 'ymax']].values
            label_names = image_data['class'].values 

        # From RGBA to RGB
        if x.shape[-1] == 4:
            from skimage.color import rgba2rgb
            x = rgba2rgb(x)

        # Read boxes
        try:
            boxes = torch.from_numpy(y).to(torch.float32)
        except TypeError:
            boxes = torch.tensor(y).to(torch.float32)

        # Label Mapping
        if self.mapping:
            labels = map_class_to_int(label_names, mapping=self.mapping)
        else:
            labels = label_names

        # Read labels
        try:
            labels = torch.from_numpy(labels).to(torch.int64)
        except TypeError:
            labels = torch.tensor(labels).to(torch.int64)

        # Convert format
        if self.convert_to_format == 'xyxy':
            boxes = box_convert(boxes, in_fmt='xywh', out_fmt='xyxy')  # transforms boxes from xywh to xyxy format
        elif self.convert_to_format == 'xywh':
            boxes = box_convert(boxes, in_fmt='xyxy', out_fmt='xywh')  # transforms boxes from xyxy to xywh format

        # Create target
        target = {'boxes': boxes,
                  'labels': labels}

        # Preprocessing
        target = {key: value.numpy() for key, value in target.items()}  # all tensors should be converted to np.ndarrays

        if self.transform is not None:
            x, target = self.transform(x, target)  # returns np.ndarrays

        # Typecasting
        x = torch.from_numpy(x).type(torch.float32)
        target = {key: torch.from_numpy(value) for key, value in target.items()}
        target['boxes'] = target['boxes'].type(torch.float32)
        return {'x': x, 'y': target, 'x_name': self.inputs[index], 'y_name': label_names}

    @staticmethod
    def read_images(inp):
        return imread(inp)