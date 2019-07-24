import os.path
import pickle
import random
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_numbering_dataset
from skimage import color  # require skimage
from PIL import Image
import numpy as np
import torchvision.transforms as transforms


class ColorizationDataset(BaseDataset):
    """This dataset class can load a set of natural images in RGB, and convert RGB format into (L, ab) pairs in Lab color space.

    This dataset is required by pix2pix-based colorization model ('--model colorization')
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        By default, the number of channels for input image  is 1 (L) and
        the nubmer of channels for output image is 2 (ab). The direction is from A to B
        """
        parser.set_defaults(input_nc=1, output_nc=2, direction='AtoB')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = [e[1] for e in sorted(make_numbering_dataset(self.dir, opt.max_dataset_size), key=lambda idx: idx[0])]
        assert(opt.input_nc == 1 and opt.output_nc == 2 and opt.direction == 'AtoB')
        self.transform = get_transform(self.opt, convert=False)

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
        # TODO: Val x is sequence first. Maybe batch first is better.
        return x, x_len

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - the L channel of an image
            B (tensor) - - the ab channels of the same image
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        path = self.AB_paths[index]
        im = Image.open(path).convert('RGB')
        im = self.transform(im)
        im = np.array(im)
        lab = color.rgb2lab(im).astype(np.float32)
        lab_t = transforms.ToTensor()(lab)
        A = lab_t[[0], ...] / 50.0 - 1.0
        B = lab_t[[1, 2], ...] / 110.0

        caption_idx = self.captions_per_image * index + random.randint(0, self.captions_per_image-1)
        caption, caption_len = self.get_caption(caption_idx)
        return {'A': A, 'B': B, 'A_paths': path, 'B_paths': path, "caption": caption, "caption_len": caption_len}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
