# VGG19を利用. これはすでに物体認識について学習をしたモデル。
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
