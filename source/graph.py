class Graph:
    """
    构建计算图 Graph
    """
    def __init__(self):
        self.operations = []
        self.placeholders = []
        self.variables = []

    def as_default(self):
        # 全局 Graph 存储网络中的 placeholder / variables / operations
        global _default_graph
        _default_graph = self


class Operation:
    """
    Operation 是计算图中完成计算的 node，接受 0 到多个输入节点，产出输出
    Operation 三要素：
    1. 使用输入得到输出的 compute 函数
    2. 输入数据列表 input_nodes
    3. 该节点输出的消费者（节点） consumers
    """
    def __init__(self, input_nodes):
        # 节点初始化
        self.input_nodes = input_nodes

        # 消费者节点使用该节点的输出作为输入
        self.consumers = []

        # 同时该节点也作为输入节点的消费节点
        for node in input_nodes:
            node.consumers.append(self)

        # 将该节点加入到所在的计算图中
        _default_graph.operations.append(self)

    def compute(self, *args):
        # 抽象方法，必须由子类实现
        pass


class placeholder:
    """
    用来表示一个 placeholder，用来构建静态计算图；
    一般为网络的输入数据，如 w * x + b 中的 x；
    该值在对所在的计算图进行前向传播得到结果之前，必须提供具体的值
    """
    def __init__(self):
        # 使用该节点作为输入的消费者节点列表
        self.consumers = []
        # 将该 placeholder 加入到计算图中
        _default_graph.placeholders.append(self)


class Variable:
    """
    不同于 placeholder， Variable 是 graph 中固有的参数，
    一般可以训练（trainable），如 w * x + b 中的 w 和 b
    """
    def __init__(self, initial_value=None):
        self.value = initial_value
        self.consumers = []
        # 将该 variable 加入到计算图中
        _default_graph.variables.append(self)