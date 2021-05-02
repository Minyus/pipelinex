import numpy as np
import torch
from torch.utils.data import DataLoader

from .numpy_ops import to_channel_last_arr


class ExplainModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, model, train_dataset, val_dataset):
        kwargs = self.kwargs

        train_size = kwargs.get("train_size")
        val_size = kwargs.get("val_size")
        train_data_loader_params = kwargs.get(
            "train_data_loader_params", dict(batch_size=100)
        )
        val_data_loader_params = kwargs.get(
            "val_data_loader_params", dict(batch_size=3)
        )
        if train_size:
            train_data_loader_params["batch_size"] = train_size
        if val_size:
            val_data_loader_params["batch_size"] = val_size
        output_transform = kwargs.get("output_transform")

        train_loader = DataLoader(train_dataset, **train_data_loader_params)
        val_loader = DataLoader(val_dataset, **val_data_loader_params)

        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))

        if output_transform is not None:
            model = torch.nn.Sequential(model, output_transform)
        images = _explain_pytorch_model(model, train_batch, val_batch)

        return images


def _explain_pytorch_model(
    model,  # type: torch.nn.Module
    train_batch,
    val_batch,
):
    try:
        import shap
    except ImportError:
        print("Failed to import shap.")
        return "_", "Failed to import shap."

    if isinstance(train_batch, (tuple, list)):
        train_images_tt, _ = train_batch
    else:
        train_images_tt = train_batch

    if isinstance(val_batch, (tuple, list)):
        val_images_tt, val_labels = val_batch
    else:
        val_images_tt = val_batch
        val_labels = [""] * val_images_tt.shape[0]

    e = shap.DeepExplainer(model, train_images_tt)
    shap_nchw_arr_list = e.shap_values(val_images_tt)

    shap_nhwc_arr_list = [to_channel_last_arr(s) for s in shap_nchw_arr_list]
    val_images_nhwc_arr = to_channel_last_arr(val_images_tt.numpy())

    shap.image_plot(shap_nhwc_arr_list, -val_images_nhwc_arr, show=True)

    shap_nhwc_arr = np.concatenate(shap_nhwc_arr_list, axis=0)

    shap_image_names = [
        "y_{}_x_{}_p_{}".format(val_labels[samp_i], samp_i, class_i)
        for class_i in range(len(shap_nchw_arr_list))
        for samp_i in range(len(val_images_tt))
    ]

    val_image_names = [
        "y_{}_x_{}".format(val_labels[samp_i], samp_i)
        for samp_i in range(len(val_images_tt))
    ]

    shap_nhwc_arr = Scale(lower=0, upper=255)(shap_nhwc_arr)
    val_images_nhwc_arr = Scale(lower=0, upper=255)(val_images_nhwc_arr)

    all_images_nhwc_arr = np.concatenate([shap_nhwc_arr, val_images_nhwc_arr], axis=0)
    all_names = shap_image_names + val_image_names
    assert all_images_nhwc_arr.shape[0] == len(all_names)

    images_dict = dict(images=all_images_nhwc_arr, names=all_names)

    return images_dict


class Scale:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, a):
        kwargs = self.kwargs

        lower = kwargs.get("lower")
        upper = kwargs.get("upper")
        if (lower is not None) or (upper is not None):
            max_val = a.max()
            min_val = a.min()
            upper = upper or max_val
            lower = lower or min_val
            a = (
                ((a - min_val) / (max_val - min_val)) * (upper - lower) + lower
            ).astype(np.uint8)
        return a
