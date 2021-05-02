import logging

import numpy as np
import torch

log = logging.getLogger(__name__)


class ModuleListMerge(torch.nn.Sequential):
    def forward(self, input):
        return [module.forward(input) for module in self._modules.values()]


class ModuleConcat(ModuleListMerge):
    def forward(self, input):
        tt_list = super().forward(input)
        assert len(set([tuple(list(tt.size())[2:]) for tt in tt_list])) == 1, (
            "Sizes of tensors must match except in dimension 1. "
            "\n{}\n got tensor sizes: \n{}\n".format(
                self, [tt.size() for tt in tt_list]
            )
        )
        return torch.cat(tt_list, dim=1)


def _check_size_match(self, tt_list):
    assert (
        len(set([tuple(list(tt.size())) for tt in tt_list])) == 1
    ), "Sizes of tensors must match. " "\n{}\n got tensor sizes: \n{}\n".format(
        self, [tt.size() for tt in tt_list]
    )


def element_wise_sum(tt_list):
    return torch.sum(torch.stack(tt_list), dim=0)


class ModuleSum(ModuleListMerge):
    def forward(self, input):
        tt_list = super().forward(input)
        _check_size_match(self, tt_list)
        return element_wise_sum(tt_list)


def element_wise_average(tt_list):
    return torch.mean(torch.stack(tt_list), dim=0)


class ModuleAvg(ModuleListMerge):
    def forward(self, input):
        tt_list = super().forward(input)
        _check_size_match(self, tt_list)
        return element_wise_average(tt_list)


def element_wise_prod(tt_list):
    return torch.prod(torch.stack(tt_list), dim=0)


class ModuleProd(ModuleListMerge):
    def forward(self, input):
        tt_list = super().forward(input)
        _check_size_match(self, tt_list)
        return element_wise_prod(tt_list)


class StatModule(torch.nn.Module):
    def __init__(self, dim, keepdim=False):
        if isinstance(dim, list):
            dim = tuple(dim)
        if isinstance(dim, int):
            dim = (dim,)
        assert isinstance(dim, tuple)
        self.dim = dim
        self.keepdim = keepdim
        super().__init__()


class Pool1dMixIn:
    def __init__(self, keepdim=False):
        super().__init__(dim=(2,), keepdim=keepdim)


class Pool2dMixIn:
    def __init__(self, keepdim=False):
        super().__init__(dim=(3, 2), keepdim=keepdim)


class Pool3dMixIn:
    def __init__(self, keepdim=False):
        super().__init__(dim=(4, 3, 2), keepdim=keepdim)


class TensorMean(StatModule):
    def forward(self, input):
        return torch.mean(input, dim=self.dim, keepdim=self.keepdim)


class TensorGlobalAvgPool1d(Pool1dMixIn, TensorMean):
    pass


class TensorGlobalAvgPool2d(Pool2dMixIn, TensorMean):
    pass


class TensorGlobalAvgPool3d(Pool3dMixIn, TensorMean):
    pass


class TensorSum(StatModule):
    def forward(self, input):
        return torch.sum(input, dim=self.dim, keepdim=self.keepdim)


class TensorGlobalSumPool1d(Pool1dMixIn, TensorSum):
    pass


class TensorGlobalSumPool2d(Pool2dMixIn, TensorSum):
    pass


class TensorGlobalSumPool3d(Pool3dMixIn, TensorSum):
    pass


class TensorMax(StatModule, torch.nn.Module):
    def forward(self, input):
        return tensor_max(input, dim=self.dim, keepdim=self.keepdim)


def tensor_max(input, dim, keepdim=False):
    if isinstance(dim, int):
        return torch.max(input, dim=dim, keepdim=keepdim)[0]
    else:
        if isinstance(dim, tuple):
            dim = list(dim)
        for d in dim:
            input = torch.max(input, dim=d, keepdim=keepdim)[0]
        return input


class TensorGlobalMaxPool1d(Pool1dMixIn, TensorMax):
    pass


class TensorGlobalMaxPool2d(Pool2dMixIn, TensorMax):
    pass


class TensorGlobalMaxPool3d(Pool3dMixIn, TensorMax):
    pass


class TensorMin(StatModule, torch.nn.Module):
    def forward(self, input):
        return tensor_min(input, dim=self.dim, keepdim=self.keepdim)


def tensor_min(input, dim, keepdim=False):
    if isinstance(dim, int):
        return torch.min(input, dim=dim, keepdim=keepdim)[0]
    else:
        if isinstance(dim, tuple):
            dim = list(dim)
        for d in dim:
            input = torch.min(input, dim=d, keepdim=keepdim)[0]
        return input


class TensorGlobalMinPool1d(Pool1dMixIn, TensorMin):
    pass


class TensorGlobalMinPool2d(Pool2dMixIn, TensorMin):
    pass


class TensorGlobalMinPool3d(Pool3dMixIn, TensorMin):
    pass


class TensorRange(StatModule, torch.nn.Module):
    def forward(self, input):
        return tensor_max(input, dim=self.dim, keepdim=self.keepdim) - tensor_min(
            input, dim=self.dim, keepdim=self.keepdim
        )


class TensorGlobalRangePool1d(Pool1dMixIn, TensorRange):
    pass


class TensorGlobalRangePool2d(Pool2dMixIn, TensorRange):
    pass


class TensorGlobalRangePool3d(Pool3dMixIn, TensorRange):
    pass


def to_array(input):
    if not isinstance(input, (tuple, list)):
        input = [input]
    input = np.array(input)
    return input


def as_tuple(x):
    return tuple(x) if isinstance(x, (list, type(np.array))) else x


def setup_conv_params(
    kernel_size=1,
    dilation=None,
    padding=None,
    stride=None,
    raise_error=False,
    *args,
    **kwargs
):
    kwargs["kernel_size"] = as_tuple(kernel_size)
    if dilation is not None:
        kwargs["dilation"] = as_tuple(dilation)

    if padding is None:
        d = dilation or 1
        d = to_array(d)
        k = to_array(kernel_size)
        p, r = np.divmod(d * (k - 1), 2)
        if raise_error and r:
            raise ValueError(
                "Invalid combination of kernel_size: {}, dilation: {}. "
                "If dilation is odd, kernel_size must be even.".format(
                    kernel_size, dilation
                )
            )
        kwargs["padding"] = tuple(p)
    else:
        kwargs["padding"] = as_tuple(padding)

    if stride is not None:
        kwargs["stride"] = as_tuple(stride)

    return args, kwargs


batchnorm_dict = {
    "1": torch.nn.BatchNorm1d,
    "2": torch.nn.BatchNorm2d,
    "3": torch.nn.BatchNorm3d,
}


class ModuleConvWrap(torch.nn.Sequential):
    core = None

    def __init__(self, batchnorm=None, activation=None, *args, **kwargs):
        args, kwargs = setup_conv_params(*args, **kwargs)
        module = self.core(*args, **kwargs)
        modules = [module]
        if batchnorm:
            if len(args) >= 2:
                out_channels = args[1]
            else:
                out_channels = kwargs["out_channels"]
            dim_str = self.core.__name__[-2]
            batchnorm_obj = batchnorm_dict[dim_str]
            if isinstance(batchnorm, dict):
                batchnorm_module = batchnorm_obj(num_features=out_channels, **batchnorm)
            else:
                batchnorm_module = batchnorm_obj(num_features=out_channels)
            modules.append(batchnorm_module)
        if activation:
            if isinstance(activation, str):
                activation = getattr(torch.nn, activation)()
            modules.append(activation)
        super().__init__(*modules)


class TensorConv1d(ModuleConvWrap):
    core = torch.nn.Conv1d


class TensorConv2d(ModuleConvWrap):
    core = torch.nn.Conv2d


class TensorConv3d(ModuleConvWrap):
    core = torch.nn.Conv3d


class TensorMaxPool1d(ModuleConvWrap):
    core = torch.nn.MaxPool1d


class TensorMaxPool2d(ModuleConvWrap):
    core = torch.nn.MaxPool2d


class TensorMaxPool3d(ModuleConvWrap):
    core = torch.nn.MaxPool3d


class TensorAvgPool1d(ModuleConvWrap):
    core = torch.nn.AvgPool1d


class TensorAvgPool2d(ModuleConvWrap):
    core = torch.nn.AvgPool2d


class TensorAvgPool3d(ModuleConvWrap):
    core = torch.nn.AvgPool3d


class ModuleBottleneck2d(torch.nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(1, 1),
        stride=(1, 1),
        mid_channels=None,
        batch_norm=None,
        activation=None,
        **kwargs
    ):
        mid_channels = mid_channels or in_channels // 2 or 1
        batch_norm = batch_norm or TensorSkip()
        activation = activation or TensorSkip()
        super().__init__(
            TensorConv2d(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                **kwargs
            ),
            batch_norm,
            activation,
            TensorConv2d(
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=kernel_size,
                stride=stride,
                **kwargs
            ),
            batch_norm,
            activation,
            TensorConv2d(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                **kwargs
            ),
        )


class TensorSkip(torch.nn.Module):
    def forward(self, input):
        return input


class TensorIdentity(torch.nn.Module):
    def forward(self, input):
        return input


class ModuleConcatSkip(ModuleConcat):
    def __init__(self, *modules):
        super().__init__(TensorIdentity(), torch.nn.Sequential(*modules))


class ModuleSumSkip(ModuleSum):
    def __init__(self, *modules):
        super().__init__(TensorIdentity(), torch.nn.Sequential(*modules))


class TensorForward(torch.nn.Module):
    def __init__(self, func=None):
        func = func or (lambda x: x)
        assert callable(func)
        self._func = func

    def forward(self, input):
        return self._func(input)


class TensorConstantLinear(torch.nn.Module):
    def __init__(self, weight=1, bias=0):
        self.weight = weight
        self.bias = bias
        super().__init__()

    def forward(self, input):
        return self.weight * input + self.bias


class TensorExp(torch.nn.Module):
    def forward(self, input):
        return torch.exp(input)


class TensorLog(torch.nn.Module):
    def forward(self, input):
        return torch.log(input)


class TensorFlatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class TensorSqueeze(torch.nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return torch.squeeze(input, dim=self.dim)


class TensorUnsqueeze(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return torch.unsqueeze(input, dim=self.dim)


class TensorSlice(torch.nn.Module):
    def __init__(self, start=0, end=None, step=1):
        super().__init__()
        self.start = start
        self.end = end
        self.step = step

    def forward(self, input):
        return input[:, self.start : (self.end or input.shape[1]) : self.step, ...]


def step_binary(input, output_size, compare=torch.ge):
    index_1dtt = input.type(dtype=torch.long)
    h_1dtt = torch.arange(output_size)
    h_2dtt = compare(h_1dtt.reshape(1, -1), index_1dtt.reshape(-1, 1))
    return h_2dtt


class StepBinary(torch.nn.Module):
    def __init__(self, size, desc=False, compare=None, dtype=None):
        super().__init__()
        assert isinstance(size, int)
        self.out_size = size
        if compare is None:
            assert isinstance(desc, bool)
            desc_dict = {False: torch.ge, True: torch.le}
            compare = desc_dict.get(desc)
        else:
            assert not desc, "'desc' and 'compare' cannot be specified together."
        self.compare = compare
        self.dtype = dtype

    def forward(self, input):
        output = step_binary(input, self.out_size, self.compare)
        dtype = self.dtype or input.type()
        return output.type(dtype=dtype)


class TensorNearestPad(torch.nn.Module):
    def __init__(self, lower=1, upper=1):
        super().__init__()
        assert isinstance(lower, int) and lower >= 0
        assert isinstance(upper, int) and upper >= 0
        self.lower = lower
        self.upper = upper

    def forward(self, input):
        return torch.cat(
            [
                input[:, :1].expand(-1, self.lower),
                input,
                input[:, -1:].expand(-1, self.upper),
            ],
            dim=1,
        )


class TensorCumsum(torch.nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return torch.cumsum(input, dim=self.dim)


class TensorClamp(torch.nn.Module):
    def __init__(self, min=None, max=None):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, input):
        return torch.clamp(input, min=self.min, max=self.max)


class TensorClampMax(torch.nn.Module):
    def __init__(self, max=None):
        super().__init__()
        self.max = max

    def forward(self, input):
        return torch.clamp_max(input, max=self.max)


class TensorClampMin(torch.nn.Module):
    def __init__(self, min=None):
        super().__init__()
        self.min = min

    def forward(self, input):
        return torch.clamp_min(input, min=self.min)


class TensorProba(torch.nn.Module):
    def __init__(self, dim=1):
        self.dim = dim
        super().__init__()

    def forward(self, input):
        total = torch.sum(input, dim=self.dim, keepdim=True)
        return input / total


def nl_loss(input, *args, **kwargs):
    return torch.nn.functional.nll_loss(input.log(), *args, **kwargs)


class NLLoss(torch.nn.NLLLoss):
    """The negative likelihood loss.
    To compute Cross Entropy Loss, there are 3 options.
    NLLoss with torch.nn.Softmax
    torch.nn.NLLLoss with torch.nn.LogSoftmax
    torch.nn.CrossEntropyLoss
    """

    def forward(self, input, target):
        return super().forward(input.log(), target)


class CrossEntropyLoss2d(torch.nn.CrossEntropyLoss):
    def forward(self, input, target):
        input_hw = list(input.shape)[-2:]
        target_hw = list(target.shape)[-2:]
        if input_hw != target_hw:
            input = torch.nn.functional.interpolate(
                input, size=target_hw, mode="bilinear", align_corners=True
            )
        input_4dtt = to_channel_last_tensor(input)
        input_2dtt = input_4dtt.reshape(-1, input_4dtt.shape[-1])
        target_1dtt = target.reshape(-1)
        return super().forward(input_2dtt, target_1dtt)


_to_channel_last_dict = {3: (-2, -1, -3), 4: (0, -2, -1, -3)}


def to_channel_last_tensor(a):
    if a.ndim in {3, 4}:
        return a.permute(*_to_channel_last_dict.get(a.ndim))
    else:
        return a


_to_channel_first_dict = {3: (-1, -3, -2), 4: (0, -1, -3, -2)}


def to_channel_first_tensor(a):
    if a.ndim in {3, 4}:
        return a.permute(*_to_channel_first_dict.get(a.ndim))
    else:
        return a
