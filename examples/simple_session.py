from tensorlite.graph import *
from tensorlite.operations import *
from tensorlite.session import Session
"""
通过一下代码，构建了一个简单的计算图：
y = A * x
z = y + b

A:Variable ----->|
                 |------> y ------>|
x:placeholder -->|                 |---> z
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

# 运行计算图
session = Session()

output = session.run(z, {
    x: [1, 2]
})

print(output)