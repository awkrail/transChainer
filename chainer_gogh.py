import argparse
import os
from PIL import Image
import numpy as np
import chainer
from chainer import optimizers
import chainer.functions as F
import chainer.links as L


def subtract_mean(x):
    copy_x = x.copy()
    copy_x[0, 0, :, :] -= 120
    copy_x[0, 1, :, :] -= 120
    copy_x[0, 2, :, :] -= 120
    return copy_x


def add_mean(x0):
    x = x0.copy()
    x[0, 0, :, :] += 120
    x[0, 1, :, :] += 120
    x[0, 2, :, :] += 120
    return x


# この関数はあとで見直す.
def image_resize(img_file, width):
    gogh = Image.open(img_file)
    orig_w, orig_h = gogh.size[0], gogh.size[1]
    if orig_w>orig_h:
        new_w = width
        new_h = width*orig_h//orig_w
        gogh = np.asarray(gogh.resize((new_w,new_h)))[:, :, :3].transpose(2, 0, 1)[::-1].astype(np.float32)
        gogh = gogh.reshape((1, 3, new_h, new_w))
        print("image resized to: ", gogh.shape)
        hoge= np.zeros((1,3,width,width), dtype=np.float32)
        hoge[0, :, width-new_h:, :] = gogh[0, :, :, :]
        gogh = subtract_mean(hoge)
    else:
        new_w = width*orig_w//orig_h
        new_h = width
        gogh = np.asarray(gogh.resize((new_w,new_h)))[:, :, :3].transpose(2, 0, 1)[::-1].astype(np.float32)
        gogh = gogh.reshape((1, 3, new_h, new_w))
        print("image resized to: ", gogh.shape)
        hoge= np.zeros((1,3,width,width), dtype=np.float32)
        hoge[0, :, :, width-new_w:] = gogh[0, :, :, :]
        gogh = subtract_mean(hoge)
    return gogh, new_w, new_h


def save_image(img, width, new_w, new_h, it):
    def to_img(x):
        im = np.zeros((new_h,new_w,3))
        im[:,:,0] = x[2,:,:]
        im[:,:,1] = x[1,:,:]
        im[:,:,2] = x[0,:,:]
        def clip(a):
            return 0 if a<0 else (255 if a>255 else a)
        im = np.vectorize(clip)(im).astype(np.uint8)
        Image.fromarray(im).save(args.out_dir+"/im_%05d.png" % it)

    if args.gpu >= 0:
        img_cpu = add_mean(img.get())
    else:
        img_cpu = add_mean(img)
    if width == new_w:
        to_img(img_cpu[0, :, width-new_h:, :])
    else:
        to_img(img_cpu[0, :, :, width-new_w:])


# 物体認識用VGG19
class VGG_chainer:
    def __init__(self, alpha=[0,0,1,1], beta=[1,1,1,1]):
        from chainer.links import VGG16Layers
        print ("load model... vgg_chainer")
        self.model = VGG16Layers()
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        feature = self.model(x, layers=["conv1_2", "conv2_2", "conv3_3", "conv4_3"])
        return [feature["conv1_2"], feature["conv2_2"], feature["conv3_3"], feature["conv4_3"]]


def get_matrix(y):
    ch = y.data.shape[1]
    wd = y.data.shape[2]
    gogh_y = F.reshape(y, (ch, wd**2))
    gogh_matrix = F.matmul(gogh_y, gogh_y, transb=True)/np.float32(ch*wd**2)
    return gogh_matrix


def main():
    parser = argparse.ArgumentParser(
        description='A Neural Algorithm of Artistic Style'
    )
    parser.add_argument('--orig_img', '-i', default='orig.png',
                        help='Original image')
    parser.add_argument('--style_img', '-s', default='style.png',
                        help='Style image')
    parser.add_argument('--out_dir', '-o', default='output',
                        help='Output directory')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID(you can choose)')
    parser.add_argument('--iter', default=5000, type=int,
                        help='number of iteration')
    parser.add_argument('--lr', default=4.0, type=float,
                        help='learning rate')
    parser.add_argument('--lam', default=0.005, type=float,
                        help='original image weight / style weight ratio')
    parser.add_argument('--width', '-w', default=435, type=int,
                        help='image width, height')
    args = parser.parse_args()

    # import ipdb; ipdb.set_trace()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    # 認識済みmodelのロード
    nn = VGG_chainer()

    W = args.width
    # print(W)
    img_content, nw, nh = image_resize(args.orig_img, W)
    img_style, _, _ = image_resize(args.style_img, W)

    # chainer_goghのアルゴリズム
    mid_org = nn.forward(img_content, volatile=True)
    style_mats = [get_matrix(y) for y in nn.forward(img_style, volatile=True)]
    img_gen = np.random.uniform(-20, 20, (1, 3, W, W), dtype=np.float32)
    img_gen = chainer.links.Parameter(img_gen)
    optimizer = optimizers.Adam(alpha=args.lr)
    optimizer.setup(img_gen)

    for i in range(args.max_iter):
        img_gen.zerograds()

        x = img_gen.W
        y = nn.forward(x)
        L = np.zeros((), dtype=np.float32)

        for l in range(len(y)):
            ch = y[l].data.shape[0]
            wd = y[l].data.shape[1]
            gogh_y = F.reshape(y[l], (ch, wd**2))
            gogh_matrix = F.matmul(gogh_y, gogh_y, transb=True)/np.float32(ch*wd**2)

            L1 = np.float32(args.lam) * np.float32(nn.alpha[l])*F.mean_squared_error(y[l], mid_org[l].data)
            L2 = np.float32(nn.beta[l]) * F.mean_squared_error(gogh_matrix, style_mats[l].data)/np.float32(len(y))
            L = L1+L2

            if i%100 == 0:
                print(i, l, L1.data, L2.data)

        L.backward()
        img_gen.W.grad = x.grad
        optimizer.update()

        tmp_shape = x.data.shape
        def clip(x):
            return -120 if x<-120 else (136 if x>136 else x)
        img_gen.W.data += np.vectorize(clip)(img_gen.W.data).reshpae(tmp_shape)

        if i%50 == 0:
            save_image(img_gen.W.data, W, nw, nh, i)

if __name__ == '__main__':
    main()