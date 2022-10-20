import inspect
import re
from typing import Any, Callable, Optional, Union

import graphviz
import torch


class Variable:

    def __init__(
            self,
            data: torch.Tensor,
            parents: tuple['Variable', ...] = (),
            grad_fn: Optional[Callable[[torch.Tensor], tuple[torch.Tensor, ...]]] = None,
            name: Optional[str] = None
    ) -> None:
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data)
        self.data = data
        self.grad: Optional[torch.Tensor] = None
        self.parents = parents
        self.grad_fn = grad_fn
        self.name = name

    def __repr__(self):
        if hasattr(self.grad_fn, 'func'):
            grad_fn_repr = self.grad_fn.func.__qualname__
        elif self.grad_fn is not None:
            grad_fn_repr = self.grad_fn.__qualname__
        else:
            grad_fn_repr = 'None'
        if self.data.ndim > 0:  # tensor
            return f"{self.__class__.__name__}(shape={tuple(self.data.shape)}, grad_fn={grad_fn_repr})"
        else:  # scalar
            return f"{self.__class__.__name__}({self.data.item()}, grad_fn={grad_fn_repr})"

    def __add__(self, other: 'Variable') -> 'Variable':
        ########################################
        # TODO: implement



        # ENDTODO
        ########################################

    def __sub__(self, other: 'Variable') -> 'Variable':
        ########################################
        # TODO: implement



        # ENDTODO
        ########################################

    def __mul__(self, other: 'Variable') -> 'Variable':
        ########################################
        # TODO: implement



        # ENDTODO
        ########################################

    def __pow__(self, power, modulo=None):
        ########################################
        # TODO: implement



        # ENDTODO
        ########################################

    def predecessors(self) -> list['Variable']:
        """

        Returns:
            predecessors: Topologically sorted list of node's predecessors including self.
        """

        predecessors = []

        ########################################
        # TODO: implement



        # ENDTODO
        ########################################

        return predecessors

    def backprop(self, dout: Optional[torch.Tensor] = None) -> None:
        """
        Runs full backpropagation starting from self. Fills the grad attribute with dself/dpredecessor for all
        predecessors of self.

        Args:
            dout: Incoming gradient on self; if None, then set to tensor of ones with proper shape and dtype

        """

        ########################################
        # TODO: implement



        # ENDTODO
        ########################################

    def to_graphviz(self, show_data: bool = False) -> graphviz.Digraph:
        def get_node_info(
                node: 'Variable',
                default_name: str = '',
                show_data: bool = False,
                stack_offset: int = 2
        ) -> tuple[str, str]:
            """
            Auxiliary function for graphviz Digraph creation to display a computation graph history of some Variable.
            Function automatically determines unique identifier and label for any node that is represented by a Variable object.

            Args:
                node: Variable object
                default_name: Name that will be displayed if automatic name determination from Variable inspection fails
                show_data: Whether to include Variable's data and grad as string in returned label
                stack_offset: Magic argument to inspect. If name determination fails, try setting this to 1 or 3.

            Returns:
                node_uid: Unique identifier of node
                node_label: Text caption of the node to be displayed as label in some graphviz.Digraph
            """
            node_uid = str(id(node))
            if node.name is not None:
                return node_uid, node.name
            try:
                namespace = dict(inspect.getmembers(inspect.stack()[stack_offset][0]))["f_globals"]
                node_name = next(k for k, v in namespace.items() if v is node)
            except StopIteration:
                node_name = default_name
            if node.data.ndim > 0:  # tensor
                node_label = f"{node_name} {node.data.shape}"
                if show_data:
                    node_label += f" | data: {node.data} | grad: {node.grad}"
            else:  # scalar
                node_label = f"{node_name} = {node.data.item()}"
                if node.grad is not None:
                    node_label += f" | grad = {node.grad.item()}"
            return node_uid, node_label

        def get_func_info(node:'Variable') -> tuple[str, str]:
            """
            The function is analogous to get_node_info, but used for graph nodes that represent operations such as summation or
            multiplication. Automatically determines unique identifier and display label.
            It will check input node's grad_fn attribute and infer the operation name of which the node is a result.
            Args:
                node: Variable object

            Returns:
                func_uid: Unique identifier of node
                func_label: Text caption of the node to be displayed as label in some graphviz.Digraph
            """
            func_uid = str(id(node)) + '.grad_fn'
            if hasattr(node.grad_fn, 'func'):
                func_label = node.grad_fn.func.__qualname__.removesuffix('.backward')
            else:
                func_label = re.search('__(\w+)__', node.grad_fn.__qualname__).group(1)
            return func_uid, func_label

        torch.set_printoptions(profile='short', threshold=6, edgeitems=1)
        dot = graphviz.Digraph(graph_attr={'rankdir': 'LR'})

        ########################################
        # TODO: implement



        # ENDTODO
        ########################################

        return dot
