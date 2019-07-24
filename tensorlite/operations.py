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

"""
通过一下代码，构建了一个简单的计算图：
y = A * x
z = y + b

A:Variable    ->|
                |------> y ------>|
x:placeholder ->|                 |---> z
                    b:Variable -->|
                    
A.consumers = [y]
x.consumers = [y]

y.input_nodes = [A, x]
y.consumers = [z]

b.consumers = [z]

z.input_nodes = [y, b]
z.consumers = []
"""

# 创建计算图
# Graph().as_default()

# variables
# A = Variable([[1, 0], [0, -1]])
# b = Variable([1, 1])

# placeholder
# x = placeholder()

# 中间节点 y
# y = matmul(A, x)

# Operation 的输出节点 z
# z = add(y, b)