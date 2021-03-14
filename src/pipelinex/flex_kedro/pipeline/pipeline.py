import kedro
from .sub_pipeline import SubPipeline
from typing import Callable, Union, List, Iterable  # NOQA


class FlexiblePipeline(kedro.pipeline.Pipeline):
    def __init__(
        self,
        nodes,  # type: Iterable[Union[SubPipeline, kedro.pipeline.Pipeline, kedro.pipeline.node.Node]]
        *,
        parameters_in_inputs=False,  # type: bool
        module="",  # type: str
        decorator=[],  # type: Union[Callable, List[Callable]]
        **kwargs
    ):

        for i, node in enumerate(nodes):

            assert node is not None, "Node {}: is empty.".format(i)
            if isinstance(node, dict):
                assert (
                    "inputs" in node
                ), "Node {} ({}): is missing 'inputs' key.".format(i, node)
                assert (
                    "outputs" in node
                ), "Node {} ({}): is missing 'outputs' key.".format(i, node)

                if parameters_in_inputs:
                    inputs = node.get("inputs")
                    inputs = inputs if isinstance(inputs, list) else [inputs]
                    if not ("parameters" in inputs):
                        node["inputs"] = inputs + ["parameters"]

                node.setdefault("module", module)

                node.setdefault("decorator", [])
                if not isinstance(node["decorator"], list):
                    node["decorator"] = [node["decorator"]]

                decorator = decorator or []
                if not isinstance(decorator, list):
                    decorator = [decorator]

                node["decorator"] = decorator + node["decorator"]

                nodes[i] = SubPipeline(**node)

        super().__init__(nodes, **kwargs)
