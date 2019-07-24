"""
Session 对象完成对一个具体的计算图的推演，
在某个 Operation 实例上计算该 operation 的输出。

由于每一个 operation 的输出依赖于其 input_nodes，
因此在计算该 operation 的输出前，必须获取其输入节点的值。

该过程通过树的前序遍历实现。
"""
import numpy as np
from tensorlite.graph import *


class Session:
    """
    Session 对一个具体的计算图进行推演
    """

    def run(self, operation, feed_dict={}):
        """
        计算一个 Operation 实例的输出
        :param operation: 需要计算输出的操作
        :param feed_dict: 需用用到的 placeholder 的具体值
        :return:
        """
        postorder_nodes = self._traversal_postorder(operation)

        for node in postorder_nodes:

            if type(node) == placeholder:
                # 将该节点树输出设置为 feed_dict 中相应的值
                node.output = feed_dict[node]
            elif type(node) == Variable:
                # 将该节点的输出设置为变量初始化值
                node.output = node.value
            elif isinstance(node, Operation):
                # 获取其 input_nodes 中每个 node 的值，并用于计算其输出
                node.inputs = [input_node.output for input_node in node.input_nodes]
                node.output = node.compute(*node.inputs)

            if type(node.output) == list:
                node.output = np.array(node.output)

        return operation.output


    @staticmethod
    def _traversal_postorder(operation):
        """
        获得某欲计算的 operation 节点的后序遍历节点(循环实现)
        :param operation: 欲计算节点
        :return: 该节点的后序遍历节点列表
        """
        postorder_nodes, stack = [], [operation]
        pre = None

        while stack:
            root = stack[-1]
            # 如果当前 root 为 Variable or placeholder
            # 或者前一个访问的节点 pre 为 root.input_nodes 中的节点
            # 则访问该节点
            # 否则先方位其 input_nodes
            if not isinstance(root, Operation) or (isinstance(root, Operation) and (pre in root.input_nodes)):
                postorder_nodes.append(root)
                pre = root
                stack.pop()
            else:
                for node in root.input_nodes:
                    stack.append(node)
        return postorder_nodes
