import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from skimage import measure

"""
  2D Image im_show in one graph 
"""

COLORS = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'white']


def show_graphs(imgs, titles=None, figsize=(5, 5), bbox=[], colors=[], show_type='gray'):
    """  Show images in a grid manner. it will automatically get a almost squared grid to show these images.

    :param imgs: input images which dim ranges in (4, 3, 2), but only the first image (HxW) can be showed
    :param titles: [str, ...], the title for every image
    :param figsize:  specify the output figure size
    :param bbox:  a list of ((min_x, max_x), (min_y, max_y))
    :param colors: a list of string of colors which length is the same as bbox
    """
    col = np.ceil(np.sqrt(len(imgs)))
    show_graph_with_col(imgs, max_cols=col, titles=titles, show=True, figsize=figsize, bbox=bbox, colors=colors, show_type=show_type)


def save_graphs(imgs, titles=None, filename='1.png', max_cols=None, figsize=(5, 5), bbox=[], colors=[], show_type='gray'):
    """  Save images in a grid manner. it will automatically get a almost squared grid to show these images.

    :param imgs: input images which dim ranges in (4, 3, 2), but only the first image (HxW) can be showed
    :param max_cols: int, max column of grid. if not specified, it will aumatically calculate the max col.
    :param titles: [str, ...], the title for every image
    :param filename: str, if save image, specify the path
    :param figsize:  specify the output figure size
    :param bbox:  a list of ((min_x, max_x), (min_y, max_y))
    :param colors: a list of string of colors which length is the same as bbox
    """
    if max_cols is None:
        max_cols = np.ceil(np.sqrt(len(imgs)))
    show_graph_with_col(imgs, max_cols=max_cols, titles=titles, show=False, filename=filename, figsize=figsize,
                        bbox=bbox, colors=colors, show_type=show_type)


def show_graph_with_col(imgs, max_cols, titles=None, show=True, filename=None, figsize=(5, 5),
                        bbox=[], colors=[], show_type='gray'):
    """ Show images in a grid manner.

    :param imgs: assume shape with [N, C, D, H, W], [N, C, H, W], [C, H, W], [N, H, W], [H, W]
             input images which dim ranges in (4, 3, 2), but only the first image (HxW) can be showed
    :param max_cols: int, max column of grid.
    :param titles: [str, ...], the title for every image
    :param show:  True or False, show or save image
    :param filename: str, if save image, specify the path
    :param figsize:  specify the output figure size
    :param bbox:  a list of ((min_x, max_x), (min_y, max_y))
    :param colors: a list of string of colors which length is the same as bbox
    """
    """
    Check size and type
    """
    if len(imgs) == 0:
        return

    length = len(imgs)
    if length < max_cols:
        max_cols = length

    img = imgs[0]
    if isinstance(img, np.ndarray):
        shape = img.shape
    elif isinstance(img, torch.Tensor):
        shape = img.size()
    else:
        raise Exception("Unknown type of imgs : {}".format(type(imgs)))
    assert 2 <= len(shape) <= 5, 'Error shape : {}'.format(shape)

    """
    Plot graph
    """
    fig = plt.figure(figsize=figsize)
    max_line = np.ceil(length / max_cols)
    for i in range(1, length + 1):
        ax = fig.add_subplot(max_line, max_cols, i)
        ax.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        if titles is not None:
            ax.set_title(titles[i - 1])

        img = imgs[i - 1]
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()

        img = img.copy()
        img[img==-1] = 0
        color = False
        shape = img.shape
        if len(shape) == 5:
            # maybe colored image
            if shape[1] == 3:
                color = True
                img = img[0, :, 0, :, :]
            else:
                img = img[0, 0, 0, :, :]
        if len(shape) == 4:
            if shape[1] == 3:
                color = True
                img = img[0]
            else:
                img = img[0, 0]
        elif len(shape) == 3:
            if shape[0] == 3:
                color = True
            else:
                img = img[0]

        if color:
            # normalized image
            if img.min() < 0:
                img = ((img * np.array([.229, .224, .225]).reshape(3, 1, 1) +
                         np.array([.485, .456, .406]).reshape(3, 1, 1)) * 255).astype(np.int32)
            img = img.transpose((1, 2, 0)).astype(np.int32)
            ax.imshow(img)
        else:
            if show_type == 'gray' or show_type == 'hot' or show_type is None:
                ax.imshow(img, cmap=show_type)
            elif show_type[:4] == 'hot_':
                vmin = int(show_type[4])
                vmax = int(show_type[5])
                ax.imshow(img, cmap=show_type[:3], vmin = vmin, vmax = vmax)
            else:
                ax.imshow(img, cmap=show_type)

        for i, box in enumerate(bbox):
            (min_x, max_x), (min_y, max_y) = box
            if len(colors) == len(bbox):
                color = colors[i]
            else:
                color = COLORS[i % len(COLORS)]
            rect = plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, fill=False, edgecolor=color, linewidth=1)
            ax.add_patch(rect)

    plt.subplots_adjust(wspace=0, hspace=0)
    if show:
        plt.show()
    else:
        plt.savefig(filename, bbox_inches='tight')


def torch_im_show_3d(input, label, logits, name='img', should_sigmoid=True):
    """ Show 3d images of the (input, label, logits). If should_sigmoid is true,
    it will perform sigmoid or softmax first to get final output.
    :param input: torch.Tensor with shape [N, C, D, H, W]
    :param label: torch.Tensor with shape [N, 1, D, H, W]
    :param logits: torch.Tensor with shape [N, C, D, H, W] if should_sigmoid is true,
                   else [N, D, H, W]
    :param name: the data name of input, normally the filename should be passed in.
    :param should_sigmoid: whether to perform sigmoid or softmax
    """
    depth = label.size()[2]
    for i in range(depth):
        if not should_sigmoid:
            torch_im_show(input[:, :, i, :, :], label[:, :, i, :, :], logits[:, i, :, :], name, should_sigmoid)
        else:
            torch_im_show(input[:, :, i, :, :], label[:, :, i, :, :], logits[:, :, i, :, :], name, should_sigmoid)


def torch_im_show(input, label, logits, name='img', should_sigmoid=True):
    """ Only show one image of the (input, label, logits). If should_sigmoid is true,
    it will perform sigmoid or softmax first to get final output.
    :param input: torch.Tensor with shape [N, C, H, W]
    :param label: torch.Tensor with shape [N, 1, H, W]
    :param logits: torch.Tensor with shape [N, C, H, W] if should_sigmoid is true,
                   else [N, H, W]
    :param name: the data name of input, normally the filename should be passed in.
    :param should_sigmoid: whether to perform sigmoid or softmax
    """
    if isinstance(logits, tuple) or isinstance(logits, list):
        logits = logits[-1]

    # get prediction
    if should_sigmoid:
        pred = get_prediction(logits)
    else:
        pred = logits
    # to numpy
    input = input.cpu().detach().numpy()
    pred = pred.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    # reduce the batch and channel dim. Only input H and W dim.
    if input.shape[1] == 3:
        input = input[0]
    else:
        input = input[0, 0]
    show_graphs([input, pred[0], label[0, 0]], [name, 'pred', 'mask'])


def show_landmark_2d(mask, landmarks):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    l = mlines.Line2D((landmarks[0, 0], landmarks[1, 0]), (landmarks[0, 1], landmarks[1, 1]))
    ax.add_line(l)
    plt.imshow(mask, cmap='gray')
    plt.show()


def show_landmark_3d(landmarks):
    x_start = landmarks[::2, 0]
    x_end   = landmarks[1::2, 0]
    y_start = landmarks[::2, 1]
    y_end   = landmarks[1::2, 1]
    z_start = landmarks[::2, 2]
    z_end   = landmarks[1::2, 1]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(x_start)):
        ax.plot([x_start[i], x_end[i]], [y_start[i], y_end[i]], zs=[z_start[i], z_end[i]])
    plt.show()


def show_3d(mask):
    verts, faces, _, _ = measure.marching_cubes_lewiner(mask, 0, spacing=(0.1, 0.1, 0.1))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap='Spectral', lw=1)
    plt.show()


def show_segmentation(mask):
    z, y, x = np.where(mask)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='r', marker='o')
    plt.show()


def label_overlay_torch(img, label, save_path=None, color=(1.0, 0., 0.), fill=True, show_type='gray'):
    H, W = img.size()[2:]
    label = F.interpolate(label.type(torch.float32), (H, W), mode='nearest')
    img = (img.detach().cpu().numpy()[0, 0] * 255).astype(np.int32)
    label = label.detach().cpu().numpy()[0, 0].astype(np.uint8)
    label_overlay(img, label, save_path, color, fill, show_type)


def label_overlay(img, label, save_path=None, color=(1.0, 0., 0.), fill=False, show_type='gray'):
    """
    :param img:   (ndarray, 3xhxw, or hxw)
    :param label: (ndarray, hxw)
    :return:
    """
    if len(img.shape) == 3:
        img = ((img * np.array([.229, .224, .225]).reshape(3, 1, 1) +
                np.array([.485, .456, .406]).reshape(3, 1, 1)) * 255)
        img = img.transpose(1, 2, 0).astype(np.int32)
    label = (label>0).astype(np.uint8)

    fig = plt.figure(frameon=False, figsize=(5, 5))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis('off')
    fig.add_axes(ax)
    ax.imshow(img, cmap=show_type)

    import cv2
    from matplotlib.patches import Polygon

    contour, hier = cv2.findContours(label.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    for c in contour:
        ax.add_patch(
            Polygon(
                c.reshape((-1, 2)),
                fill=fill, facecolor=color, edgecolor='r', linewidth=2.0, alpha=0.5
            )
        )
    if save_path is None:
        plt.imshow(img, cmap='gray')
        plt.show()
    else:
        fig.savefig(save_path)
    plt.close('all')


class ImageOverlay(object):
    def __init__(self, img, cmap='gray'):
        assert isinstance(img, np.ndarray), len(img.shape) in [2, 3]

        if len(img.shape) == 3:
            img = ((img * np.array([.229, .224, .225]).reshape(3, 1, 1) +
                    np.array([.485, .456, .406]).reshape(3, 1, 1)) * 255)
            img = img.transpose(1, 2, 0).astype(np.int32)

        fig = plt.figure(frameon=False, figsize=(5, 5))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.axis('off')
        fig.add_axes(ax)
        ax.imshow(img, cmap=cmap)

        self.fig = fig
        self.ax = ax

    def overlay(self, mask, color=(1., 0., 0.), edgecolor='r', fill=False, linewidth=2.0, alpha=0.5):
        assert isinstance(mask, np.ndarray), len(mask.shape) == 2

        import cv2
        from matplotlib.patches import Polygon

        mask = (mask > 0).astype(np.uint8)
        # _, contour, hier = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        contour, hier = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        for c in contour:
            self.ax.add_patch(
                Polygon(
                    c.reshape((-1, 2)),
                    fill=fill, facecolor=color, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha,
                )
            )
        return self

    def overlay_hole(self, mask, color=(1., 0., 0.), edgecolor='r', fill=False, linewidth=2.0, alpha=0.5):
        import cv2
        from matplotlib.path import Path

        mask = self.to_numpy(mask)
        mask = (mask > 0).astype(np.uint8)
        contour, hier = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        path_points = []
        path_move = []
        for c in contour:
            c = c.reshape(-1, 2)
            for i, p in enumerate(c):
                path_points.append(p)
                if i == 0:
                    path_move.append(Path.MOVETO)
                elif i == len(c)-1:
                    path_move.append(Path.CLOSEPOLY)
                else:
                    path_move.append(Path.LINETO)
        from matplotlib.patches import PathPatch
        patch = PathPatch(Path(path_points, path_move), fill=fill, facecolor=color, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha)
        self.ax.add_patch(patch)
        return self

    def show(self):
        plt.show()
        return self

    def save(self, save_path):
        self.fig.savefig(save_path)
        return self


def show_tsne(X, labels):
    from sklearn.manifold.t_sne import TSNE
    tsne = TSNE(n_components=2)
    tsne_X = tsne.fit_transform(X)

    tsne_X = (tsne_X - tsne_X.min()) / (tsne_X.max() - tsne_X.min())

    for i in range(tsne_X.shape[0]):
        plt.scatter(tsne_X[i, 0], tsne_X[i, 1], color=plt.cm.Set1(labels[i]), )
    plt.show()


if __name__ == '__main__':
    # img_p = '/data/datasets/new_kidney/train/D0013330343.jpg'
    # mask_p = '/data/datasets/new_kidney/train/D0013330343_1.bmp'
    # import skimage.io as skio
    # img = skio.imread(img_p, as_gray=True)
    # mask = skio.imread(mask_p, as_gray=True)
    # label_overlay(img, mask)

    from sklearn import datasets
    digits = datasets.load_digits(n_class=6)
    X = digits.data
    y = digits.target
    show_tsne(X, y)
