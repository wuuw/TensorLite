from tensorlite.graph import Operation, Graph, Variable, placeholder
import numpy as np

class add(Operation):
    """
    进行 element-wise 加法计算
    """
    def __init__(self, x, y):
        """
        使用 Operation 类初始化一个 node
        :param x: node x
        :param y: node y
        """
        super(add, self).__init__([x, y])

    def compute(self, x_value, y_value):
        """
        :param x_value: 加数
        :param y_value: 加数
        :return: element-wise sum
        """
        return x_value + y_value


class matmul(Operation):
    """
    scaler/vector/matrix 求积
    """
    def __init__(self, x, y):
        super(matmul, self).__init__([x, y])

    def compute(self, x_value, y_value):
        return np.dot(x_value, y_value)


# 创建计算图
Graph().as_default()

# variables
A = Variable([[1, 0], [0, -1]])
b = Variable([1, 1])

# placeholder
x = placeholder()

# 中间节点 y
y = matmul(A, x)

# Operation 的输出节点 z
z = add(y, b)