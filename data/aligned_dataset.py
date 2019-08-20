import os.path
import pickle
import random
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_numbering_dataset
import numpy as np
from PIL import Image


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = [
            e[1] for e in sorted(make_numbering_dataset(self.dir_AB, opt.max_dataset_size), key=lambda idx: idx[0])]
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

        with open(opt.captions, 'rb') as f:
            x = pickle.load(f)
            train_captions, test_captions = x[0], x[1]
            self.captions = train_captions if opt.phase == "train" else test_captions
            self.ixtoword, self.wordtoix = x[2], x[3]
            del x, train_captions, test_captions
            self.n_words = len(self.ixtoword)
            print('Load from: ', opt.captions)
        self.captions_per_image = opt.captions_per_image
        self.text_words_num = opt.text_words_num

    def get_caption(self, sent_ix):
        # a list of indices for a sentence
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros(self.text_words_num, dtype='int64')
        x_len = num_words
        if num_words <= self.text_words_num:
            x[:num_words] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:self.text_words_num]
            ix = np.sort(ix)
            x = sent_caption[ix]
            x_len = self.text_words_num
        return x, x_len

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        # split AB image into A and B
        w, h = AB.size
        if w > h:
            w2 = int(w / 2)
            A = AB.crop((0, 0, w2, h))
            B = AB.crop((w2, 0, w, h))
        else:
            A = AB
            B = AB

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        caption_idx = self.captions_per_image * index + random.randint(0, self.captions_per_image - 1)
        caption, caption_len = self.get_caption(caption_idx)
        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path,
                "caption": caption, "caption_len": caption_len}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
