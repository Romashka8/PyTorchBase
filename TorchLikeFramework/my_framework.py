import numpy as np


class Tensor(object):
    def __init__(self, data, autograd=False, creators=None, creation_op=None, id=None):
        """
        :param data: Tensor data
        :param creators: Other tensors, used in creation of current tensor
        :param creation_op: Others tensors operation
        :param autograd: calc gradient automatic or not
        :param id: self tensor id(using in calculations graph)
        """
        self.data = np.array(data)
        self.autograd = autograd
        self.grad = None
        if id is None:
            self.id = np.random.randint(0, 100000)
        else:
            self.id = id
        # creators
        self.creation_op = creation_op
        self.creators = creators
        self.children = {}
        # add new tensor into creator's tensors children
        if creators is not None:
            for c in creators:
                if self.id not in c.children:
                    c.children[self.id] = 1
                else:
                    c.children[self.id] += 1

    # check if tensor got gradients from children
    def all_children_grads_accounted_for(self):
        for id, cnt in self.children.items():
            if cnt != 0:
                return False
        return True

    # recursive backpropagation
    def backward(self, grad=None, grad_origin=None):
        # initialize tensors gradient
        if self.autograd:
            if grad is None:
                grad = Tensor(np.ones_like(self.data, dtype=np.float32))

            if grad_origin is not None:
                # check if we still can do backprop(collect they final gradient)
                if self.children[grad_origin.id] == 0:
                    raise Exception('cannot backprop more than once')
                else:
                    self.children[grad_origin.id] -= 1
            # collect grads from children
            if self.grad is None:
                self.grad = grad
            else:
                self.grad += grad
            # grads must not have grads for they own!
            assert grad.autograd == False
            # only continue backpropping if there's something to
            # backprop into and if all gradients (from children)
            # are accounted for override waiting for children if
            # "backprop" was called on this variable directly
            if self.creators is not None\
                    and self.all_children_grads_accounted_for() or grad_origin is None:

                if self.creation_op == 'add':
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad, self)

                if self.creation_op == 'sub':
                    self.creators[0].backward(Tensor(self.grad.data), self)
                    self.creators[1].backward(Tensor(self.grad.__neg__().data), self)

                if self.creation_op == 'mul':
                    new = self.grad * self.creators[1]
                    self.creators[0].backward(new, self)
                    new = self.grad * self.creators[0]
                    self.creators[1].backward(new, self)

                if self.creation_op == 'mm':
                    c0 = self.creators[0]
                    c1 = self.creators[1]
                    new = self.grad.mm(c1.transpose())
                    c0.backward(new)
                    new = self.grad.transpose().mm(c0).transpose()
                    c1.backward(new)

                if self.creation_op == 'transpose':
                    self.creators[0].backward(self.grad.transpose())

                if 'sum_' in self.creation_op:
                    dim = int(self.creation_op.split('_')[1])
                    self.creators[0].backward(self.grad.expand(dim,
                                                               self.creators[0].data.shape[dim]))

                if 'expand_' in self.creation_op:
                    dim = int(self.creation_op.split('_')[1])
                    self.creators[0].backward(self.grad.sum(dim))

                if self.creation_op == 'neg':
                    self.creators[0].backward(self.grad.__neg__())

    def __add__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data + other.data,
                          autograd=True,
                          creators=[self, other],
                          creation_op='add')
        return Tensor(self.data + other.data)

    def __neg__(self):
        if self.autograd:
            return Tensor(self.data * -1,
                          autograd=True,
                          creators=[self],
                          creation_op='neg')
        return Tensor(self.data * -1)

    def __sub__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data - other.data,
                          autograd=True,
                          creators=[self, other],
                          creation_op='sub')
        return Tensor(self.data - other.data)

    def __mul__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data * other.data,
                          autograd=True,
                          creators=[self, other],
                          creation_op='mul')
        return Tensor(self.data * other.data)

    def sum(self, dim):
        if self.autograd:
            return Tensor(self.data.sum(dim),
                          autograd=True,
                          creators=[self],
                          creation_op='sum_' + str(dim))
        return Tensor(self.data.sum(dim))

    def expand(self, dim, copies):

        trans_cmd = list(range(0, len(self.data.shape)))
        trans_cmd.insert(dim, len(self.data.shape))
        new_data = self.data.repeat(copies).reshape(list(self.data.shape) + [copies]).transpose(trans_cmd)

        if self.autograd:
            return Tensor(new_data,
                          autograd=True,
                          creators=[self],
                          creation_op='expand_' + str(dim))
        return Tensor(new_data)

    def transpose(self):
        if self.autograd:
            return Tensor(self.data.transpose(),
                          autograd=True,
                          creators=[self],
                          creation_op='transpose')
        return Tensor(self.data.transpose())

    def mm(self, x):
        if self.autograd:
            return Tensor(self.data.dot(x.data),
                          autograd=True,
                          creators=[self, x],
                          creation_op='mm')
        return Tensor(self.data.dot(x.data))

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())


# Adding SGD optimizer
class SGD(object):
    def __init__(self, parameters, alpha=0.1):
        """
        :param parameters: optimizing parameters(weights in neural network layers)
        :param alpha: SDG coef
        """
        self.parameters = parameters
        self.alpha = alpha

    def zero(self):
        for p in self.parameters:
            p.grad.data *= 0

    def step(self, zero=True):
        """
        :param zero: we need to zero grads for correct working
        """
        for p in self.parameters:
            p.data -= p.grad.data * self.alpha
            if zero:
                p.grad.data *= 0


# adding base layer class
class Layer(object):
    def __init__(self):
        self.parameters = list()

    def get_parameters(self):
        return self.parameters


class Linear(Layer):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        w = np.random.randn(n_inputs, n_outputs) * np.sqrt(2.0 / n_inputs)
        self.weight = Tensor(w, autograd=True)
        self.bias = Tensor(np.zeros(n_outputs), autograd=True)

        self.parameters.append(self.weight)
        self.parameters.append(self.bias)

    def forward(self, input):
        return input.mm(self.weight) + self.bias.expand(0, len(input.data))


# Sequential module implementation
class Sequential(Layer):
    def __init__(self, layers=list()):
        super().__init__()

        self.layers = layers

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def get_parameters(self):
        params = list()
        for l in self.layers:
            params += l.get_parameters()
        return params


if __name__ == '__main__':
    a = Tensor([1, 2, 3, 4, 5], autograd=True)
    b = Tensor([2, 2, 2, 2, 2], autograd=True)
    c = Tensor([5, 4, 3, 2, 1], autograd=True)

    d = a - b
    e = b + c
    f = d + e

    f.backward(Tensor(np.array([1, 1, 1, 1, 1])))

    print(b.grad.data == np.array([2, 2, 2, 2, 2]))
