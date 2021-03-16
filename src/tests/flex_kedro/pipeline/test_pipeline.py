from functools import wraps
from typing import Callable

import kedro

import pipelinex


def task_func1(a):
    return a


def task_func2(a):
    return a


def task_func3(a):
    return a


def stringify(obj):
    print("\n" + obj.__str__())
    print("\n" + obj.to_json())
    s = obj.to_json()
    return s.replace("Flexible", "")


def assert_eq(obj1, obj2):
    assert stringify(obj1) == stringify(obj2)


def task_deco(func: Callable) -> Callable:
    @wraps(func)
    def _task_deco(a):
        a += 10
        result = func(a)
        result += 100
        return result

    return _task_deco


def test_dict():
    fp = pipelinex.FlexiblePipeline(
        nodes=[dict(func=task_func1, inputs="my_input", outputs="my_output")]
    )
    kp = kedro.pipeline.Pipeline(
        nodes=[
            kedro.pipeline.node(func=task_func1, inputs="my_input", outputs="my_output")
        ]
    )
    assert_eq(fp, kp)


def test_sequential():
    fp = pipelinex.FlexiblePipeline(
        nodes=[
            dict(
                func=[task_func1, task_func2, task_func3],
                inputs="my_input",
                outputs="my_output",
            )
        ]
    )
    kp = kedro.pipeline.Pipeline(
        nodes=[
            kedro.pipeline.node(
                func=task_func1, inputs="my_input", outputs="my_output__001"
            ),
            kedro.pipeline.node(
                func=task_func2, inputs="my_output__001", outputs="my_output__002"
            ),
            kedro.pipeline.node(
                func=task_func3, inputs="my_output__002", outputs="my_output"
            ),
        ]
    )
    assert_eq(fp, kp)


def test_decorator():
    if kedro.__version__[:5] not in {"0.16.", "0.17."}:
        return
    fp = pipelinex.FlexiblePipeline(
        nodes=[
            kedro.pipeline.node(func=task_func1, inputs="my_input", outputs="my_output")
        ],
        decorator=[task_deco, task_deco],
    )
    kp = kedro.pipeline.Pipeline(
        nodes=[
            kedro.pipeline.node(func=task_func1, inputs="my_input", outputs="my_output")
        ]
    ).decorate(task_deco, task_deco)
    assert_eq(fp, kp)
