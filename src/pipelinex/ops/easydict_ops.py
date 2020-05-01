from easydict import EasyDict


def to_easydict(
    d={},  # type: dict
):
    return EasyDict(d)
