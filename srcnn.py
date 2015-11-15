from chainer import FunctionSet, Variable
import chainer.functions as F

class SRCNN(FunctionSet):
    """Super Resolution Convolutional Neural Network"""

    def __init__(self, c=3, n1=64, n2=32, f1=9, f2=3, f3=5):
        super(SRCNN, self).__init__(
            layer1 = F.Convolution2D( c, n1, f1, stride=1),
            layer2 = F.Convolution2D(n1, n2, f2, stride=1),
            layer3 = F.Convolution2D(n2,  c, f3, stride=1)
        )
        self.c  = c
        self.n1 = n1
        self.n2 = n2
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3

    def forward(self, x_data, y_data, train=True):
        x = Variable(x_data, volatile=not train)
        t = Variable(y_data, volatile=not train)

        h = F.relu(self.layer1(x))
        h = F.relu(self.layer2(h))
        h = self.layer3(h)

        return F.mean_squared_error(h, t)
