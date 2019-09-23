import warnings

import cv2
import numpy as np
import torch
from torch.utils.data import TensorDataset
from tqdm.auto import tqdm


class Tetrominoes:
    """
    Dataset containing color images of multiple tetris pieces.

    Arguments:
        lim_angles (list of two floats): Lower and upper bound for angle
        sample_angles ('continuous' or 'discrete'): if 'discrete' uses evenly spaced numbers between lim_angles,
        if 'continuous' samples
        num_angles (int):
    """

    def __init__(self, height=32, width=32,
                 lim_angles=None, num_angles=16, sample_angles='continuous',
                 lim_scales=None, num_scales=5, sample_scales='continuous',
                 lim_colors=None, num_colors=8, sample_colors='continuous',
                 lim_xs=None, num_xs=16, sample_xs='continuous',
                 lim_ys=None, num_ys=16, sample_ys='continuous',
                 shapes=None, num_train_per_shape=None, num_val_per_shape=None, num_test_per_shape=None,
                 seed=1, constraints=None,
                 num_processes=1, mode=None):

        if mode in ['id', 'ood']:
            height = 32
            width = 32
            num_angles = 16
            lim_angles = [0, 360 * (1 - 1 / num_angles)]
            num_colors = 8
            lim_colors = [0, 1 - 1 / num_colors]
            num_scales = 5
            lim_scales = [2, 5]
            num_xs = 16
            lim_xs = [lim_scales[1] * 2 - 2, width - lim_scales[1] * 2 + 1]
            num_ys = 16
            lim_ys = [lim_scales[1] * 2 - 2, height - lim_scales[1] * 2 + 1]
            shapes = [0]
            seed = 1

            if mode == 'id':
                if num_train_per_shape is None:
                    num_train_per_shape = 81920
                if num_val_per_shape is None:
                    num_val_per_shape = 10000
                num_test_per_shape = 81920
                constraints = None
            elif mode == 'ood':
                constraints = checkerboard_pattern_5d(lim_angles, lim_colors, lim_scales,
                                                      lim_xs, lim_ys, n_checker=1)

        self.height = height
        self.width = width
        if seed is not None:
            np.random.seed(seed)
        if lim_angles is None:
            lim_angles = [0, 359]
        if lim_colors is None:
            lim_colors = [0, 1 - 1 / num_colors]
        if lim_scales is None:
            lim_scales = [1.2, 3]
        if lim_xs is None:
            lim_xs = [lim_scales[1] - 2, width - lim_scales[1] + 1]
        if lim_ys is None:
            lim_ys = [lim_scales[1] - 2, height - lim_scales[1] + 1]
        if shapes is None:
            shapes = [0]
        num_samples_per_shape = num_train_per_shape + num_val_per_shape + num_test_per_shape
        train_ratio = 1 - num_test_per_shape / num_samples_per_shape

        # features is a dict that contains a list for every feature with following structure:
        # [is_discrete, num_features, lim_features, features_array]
        features = dict()
        features['angles'] = [sample_angles == 'discrete', num_angles, lim_angles,
                              np.linspace(lim_angles[0], lim_angles[1],
                                          num_angles) if sample_angles == 'discrete' else None]
        features['colors'] = [sample_colors == 'discrete', num_colors, lim_colors,
                              np.linspace(lim_colors[0], lim_colors[1],
                                          num_colors) if sample_colors == 'discrete' else None]
        features['scales'] = [sample_scales == 'discrete', num_scales, lim_scales,
                              np.linspace(lim_scales[0], lim_scales[1],
                                          num_scales) if sample_scales == 'discrete' else None]
        features['xs'] = [sample_xs == 'discrete', num_xs, lim_xs,
                          np.linspace(lim_xs[0], lim_xs[1], num_xs) if sample_xs == 'discrete' else None]
        features['ys'] = [sample_ys == 'discrete', num_ys, lim_ys,
                          np.linspace(lim_ys[0], lim_ys[1], num_ys) if sample_ys == 'discrete' else None]

        num_grid_points = np.multiply.reduce(
            [num_features if is_discrete else 1 for is_discrete, num_features, _, _ in features.values()])

        if np.all([is_discrete for is_discrete, _, _, _ in features.values()]):
            # if all features are discrete, number of samples are size of the cartesian product of features,
            # so num_samples_per_shape is not important
            warnings.warn('All features are discrete, omitting num_samples_per_shape')
            num_samples_per_shape = num_grid_points
        else:
            if num_samples_per_shape < num_grid_points:
                # if there is at least one continuous feature, but num_samples_per_shape is smaller than
                # size of the cartesian product of discrete features, it's ambiguous what user wants, give error
                raise ValueError(("Product of number of discrete features ({})"
                                  " is larger than num_samples_per_shape ({})").format(
                    num_grid_points, num_samples_per_shape))
            elif not (num_samples_per_shape / num_grid_points).is_integer():
                num_samples_per_shape = int(np.round(num_samples_per_shape / num_grid_points) * num_grid_points)
                warnings.warn(('num_samples_per_shape should be an integer multiple of the product of number of'
                               ' discrete features. Setting it to nearest integer multiple: {}').format(
                    num_samples_per_shape))

        num_samples = num_samples_per_shape * len(shapes)
        # placeholder is a placeholder for continuous features for meshgrid. Continuous features are not included
        # in cartesian product, so there is only one placeholder (opposed to placeholder for every continuous feature)
        placeholder = np.zeros(num_samples // num_grid_points)
        # adding shapes here
        features['shapes'] = [True, len(shapes), None, np.array(shapes)]  # limits are not important at this point
        grid = np.meshgrid(*([f for is_discrete, _, _, f in features.values() if is_discrete] + [placeholder]))
        discrete_names = [names for names, [is_discrete, _, _, _] in features.items() if is_discrete]
        for i in range(len(grid) - 1):
            grid[i] = grid[i].flatten()
            features[discrete_names[i]][3] = grid[i]
        
        # do the sampling for continuous features
        continuous_names = []
        continuous_args = []
        for name, [is_discrete, num_features, lim_features, _] in features.items():
            if not is_discrete:
                continuous_names.append(name)
                continuous_args.append([lim_features[0], lim_features[1], num_features])
        
        if len(continuous_names) > 0:
            continuous_features = stratified_uniform(*continuous_args, num_samples=num_samples)
            print(continuous_features.shape)
            for i, name in enumerate(continuous_names):
                features[name][3] = continuous_features[:, i]
        if constraints is None:
            # no constraint, do random partitioning according to train_ratio
            cut = features['angles'][3].shape[0] * train_ratio
            mask = np.random.permutation(features['angles'][3].shape[0]) < cut
        else:
            # apply constraints
            mask = constraints(features['angles'][3], features['colors'][3],
                               features['scales'][3], features['xs'][3], features['ys'][3])
        
        self.train_labels = np.stack([f[mask] for _, _, _, f in features.values()], axis=-1)
        mask = np.logical_not(mask)
        self.test_labels = np.stack([f[mask] for _, _, _, f in features.values()], axis=-1)
        val_ratio = num_val_per_shape / (num_train_per_shape + num_val_per_shape)
        cut = self.train_labels.shape[0] * val_ratio
        mask = np.random.permutation(self.train_labels.shape[0]) < cut
        self.val_labels  = self.train_labels[mask]
        self.train_labels = self.train_labels[np.logical_not(mask)]  
        if num_processes > 1:
            try:
                import multiprocessing as mp
                with mp.Pool(processes=num_processes) as pool:
                    self.train_data = pool.starmap(self.get_data_by_label, self.train_labels.tolist())
                with mp.Pool(processes=num_processes) as pool:
                    self.val_data = pool.starmap(self.get_data_by_label, self.val_labels.tolist())
                with mp.Pool(processes=num_processes) as pool:
                    self.test_data = pool.starmap(self.get_data_by_label, self.test_labels.tolist())
            except ImportError:
                warnings.warn('Error importing multiprocessing, setting num_processes=1.')
                num_processes = 1

        if num_processes == 1:
            self.train_data = []
            for i in tqdm(range(self.train_labels.shape[0]), desc='Train data'):
                self.train_data.append(self.get_data_by_label(self.train_labels[i, 0], self.train_labels[i, 1],
                                                              self.train_labels[i, 2], self.train_labels[i, 3],
                                                              self.train_labels[i, 4], self.train_labels[i, 5]))
            self.test_data = []
            for i in tqdm(range(self.test_labels.shape[0]), desc='Test data'):
                self.test_data.append(self.get_data_by_label(self.test_labels[i, 0], self.test_labels[i, 1],
                                                             self.test_labels[i, 2], self.test_labels[i, 3],
                                                             self.test_labels[i, 4], self.test_labels[i, 5]))
                                                             self.test_data = []
            self.val_data = []
            for i in tqdm(range(self.val_labels.shape[0]), desc='Val data'):
                self.val_data.append(self.get_data_by_label(self.val_labels[i, 0], self.val_labels[i, 1],
                                                            self.val_labels[i, 2], self.val_labels[i, 3],
                                                            self.val_labels[i, 4], self.val_labels[i, 5]))

        self.train_data = torch.tensor(np.stack(self.train_data, axis=0),
                                       dtype=torch.float).permute(0, 3, 1, 2).reshape(-1, height * width * 3)
        self.test_data = torch.tensor(np.stack(self.test_data, axis=0),
                                      dtype=torch.float).permute(0, 3, 1, 2).reshape(-1, height * width * 3)
        self.val_data = torch.tensor(np.stack(self.val_data, axis=0),
                                      dtype=torch.float).permute(0, 3, 1, 2).reshape(-1, height * width * 3)
        self.train_labels = torch.tensor(self.train_labels)
        self.test_labels = torch.tensor(self.test_labels)
        self.val_labels = torch.tensor(self.val_labels)
        self.num_train = self.train_data.shape[0]
        self.num_val = self.val_data.shape[0]
        self.num_test = self.test_data.shape[0]

    @property
    def train_dataset(self):
        return TensorDataset(self.train_data, self.train_labels)
        
    @property
    def val_dataset(self):
        return TensorDataset(self.val_data, self.val_labels)

    @property
    def test_dataset(self):
        return TensorDataset(self.test_data, self.test_labels)

    @staticmethod
    def get_data_by_label(angle=0, color=0, scale=1, x=16, y=16, shape=0, height=32, width=32, value=1.0,
                          flag_affine=cv2.INTER_AREA, flag_resize=cv2.INTER_AREA):
        int_final_ratio = 16
        final_shape = (height, width)
        intermediate_shape = (height * int_final_ratio, width * int_final_ratio)
        if shape == 0:  # J
            tetromino = np.zeros((300, 200), dtype=np.float32)
            tetromino[:, 100:] = 1
            tetromino[200:, :] = 1
        elif shape == 1:  # L
            tetromino = np.zeros((300, 200), dtype=np.float32)
            tetromino[:, :100] = 1
            tetromino[200:, :] = 1
        elif shape == 2:  # |
            tetromino = np.ones((400, 100), dtype=np.float32)
        elif shape == 3:  # T
            tetromino = np.zeros((200, 300), dtype=np.float32)
            tetromino[:, 100:200] = 1
            tetromino[100:, :] = 1
        elif shape == 4:  # 2
            tetromino = np.zeros((200, 300), dtype=np.float32)
            tetromino[:100, :200] = 1
            tetromino[100:, 100:] = 1
        elif shape == 5:  # 5
            tetromino = np.zeros((200, 300), dtype=np.float32)
            tetromino[100:, :200] = 1
            tetromino[:100, 100:] = 1
        elif shape == 6:  # square
            tetromino = np.ones((200, 200), dtype=np.float32)
        else:
            raise ValueError("invalid shape: {}".format(shape))

        scale_ = scale / 100 * int_final_ratio
        t1 = np.eye(3)  # First translation moves center of shape to origin
        t1[0, 2] = -tetromino.shape[1] / 2
        t1[1, 2] = -tetromino.shape[0] / 2
        r = np.eye(3)  # Rotation
        r[0, 0] = scale_ * np.cos(angle * np.pi / 180)
        r[0, 1] = scale_ * np.sin(angle * np.pi / 180)
        r[1, 0] = -r[0, 1]
        r[1, 1] = r[0, 0]
        t2 = np.eye(3)  # Second translation moves rotated shape to x, y
        t2[0, 2] = int_final_ratio * (x + 0.5)
        t2[1, 2] = int_final_ratio * (y + 0.5)
        affine_mat = (t2 @ r @ t1)[:-1]

        dst = cv2.warpAffine(tetromino, affine_mat, intermediate_shape, flags=flag_affine)
        dst = cv2.resize(dst, final_shape, interpolation=flag_resize)
        dst = value * np.repeat(dst[..., np.newaxis], 3, axis=2)
        dst[..., 1] = 1
        dst[..., 0] = color * 360
        dst = cv2.cvtColor(dst, cv2.COLOR_HSV2RGB)
        dst[dst > 1] = 1
        dst[dst < 0] = 0
        return dst

    def visualize(self, num_points=100, num_cols=10, random=True, train=True):
        from matplotlib import pyplot as plt
        data = self.train_data if train else self.test_data
        labels = self.train_labels if train else self.test_labels
        p = torch.randint(0, data.shape[0], (num_points,)) if random else torch.arange(num_points)
        p = p.long()
        samples = data[p].reshape(-1, 3, self.height, self.width).permute(0, 2, 3, 1).numpy()
        labels = labels[p].numpy()
        num_rows = num_points // num_cols
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 15))
        for i, ax in enumerate(axs.flat):
            ax.imshow(samples[i])
            ax.set_title('a:{:.1f},c:{:.1f},\ns:{:.1f},x:{:.1f},\ny:{:.1f},sh:{:d}'.format(labels[i, 0], labels[i, 1],
                                                                                           labels[i, 2], labels[i, 3],
                                                                                           labels[i, 4],
                                                                                           int(labels[i, 5])))
        plt.tight_layout()


def stratified_uniform(*args, num_samples=None, shuffle=False):
    borders = [np.linspace(low, high, num_bins, endpoint=False) for [low, high, num_bins] in args]
    num_grids = np.multiply.reduce([num_bins for [_, _, num_bins] in args])
    num_samples = num_grids if num_samples is None else num_samples
    if num_samples < num_grids:
        raise ValueError('num_samples must be greater than product of num_bins of all features')
    borders = np.meshgrid(*borders)
    for i in range(len(borders)):
        borders[i] = borders[i].flatten()
    borders = np.stack(borders, axis=-1)
    assert borders.shape == (num_grids, len(args))

    samples = []
    widths = np.array([[(high - low) / num_bins for [low, high, num_bins] in args]])
    for _ in range(num_samples // num_grids):
        samples.append(borders + np.random.rand(*borders.shape) * widths)
    if num_samples % num_grids > 0:
        lows = np.array([[low for [low, _, _] in args]])
        highs = np.array([[high for [_, high, _] in args]])
        samples.append(lows + np.random.rand(num_samples % num_grids, lows.shape[-1]) * (highs - lows))
    samples = np.concatenate(samples, axis=0)
    if shuffle:
        np.random.shuffle(samples)
    return samples


def checkerboard_pattern_5d(lim_angles=None, lim_colors=None, lim_scales=None,
                            lim_xs=None, lim_ys=None, n_checker=1):
    if lim_angles is None:
        lim_angles = [0, 360]
    if lim_colors is None:
        lim_colors = [0, 1]
    if lim_scales is None:
        lim_scales = [6, 10]
    if lim_xs is None:
        lim_xs = [8, 23]
    if lim_ys is None:
        lim_ys = [8, 23]
    if n_checker < 1:
        return np.array([lambda x, y: True * np.ones(*x.shape)])

    width_a = (lim_angles[1] + 1e-9 - lim_angles[0]) / n_checker / 2
    width_c = (lim_colors[1] + 1e-9 - lim_colors[0]) / n_checker / 2
    width_s = (lim_scales[1] + 1e-9 - lim_scales[0]) / n_checker / 2
    width_x = (lim_xs[1] + 1e-9 - lim_xs[0]) / n_checker / 2
    width_y = (lim_ys[1] + 1e-9 - lim_ys[0]) / n_checker / 2

    def check_fn(a, c, s, x, y,
                 low_angles=lim_angles[0], low_colors=lim_colors[0], low_scales=lim_scales[0],
                 low_xs=lim_xs[0], low_ys=lim_ys[0],
                 width_a=width_a, width_c=width_c, width_s=width_s,
                 width_x=width_x, width_y=width_y):
        p = (1 - np.indices((n_checker * 2, n_checker * 2, n_checker * 2, n_checker * 2, n_checker * 2)).sum(
            axis=0) % 2).astype(np.bool)
        return p[((a - low_angles) / width_a).astype(np.long),
                 ((c - low_colors) / width_c).astype(np.long),
                 ((s - low_scales) / width_s).astype(np.long),
                 ((x - low_xs) / width_x).astype(np.long),
                 ((y - low_ys) / width_y).astype(np.long)]

    return check_fn
