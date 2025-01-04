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

    def __add__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data + other.data,
                          autograd=True,
                          creators=[self, other],
                          creation_op='add')
        return Tensor(self.data + other.data)

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())


if __name__ == '__main__':
    a = Tensor([1, 2, 3, 4, 5], autograd=True)
    b = Tensor([2, 2, 2, 2, 2], autograd=True)
    c = Tensor([5, 4, 3, 2, 1], autograd=True)

    d = a + b
    e = b + c
    f = d + e

    f.backward(Tensor(np.array([1, 1, 1, 1, 1])))
    print(b.grad.data == np.array([2, 2, 2, 2, 2]))
