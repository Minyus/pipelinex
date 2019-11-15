import os
from datetime import datetime, timedelta
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import ignite
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import RunningAverage
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.contrib.handlers.tqdm_logger import ProgressBar
import numpy as np
import time
from pkg_resources import parse_version
import logging

log = logging.getLogger(__name__)


class NetworkTrain:
    def __init__(
        self,
        train_params,  # type: dict
        mlflow_logging=True,  # type: bool
    ):
        self.train_params = train_params
        self.mlflow_logging = mlflow_logging

    def __call__(self, model, train_dataset, val_dataset=None, parameters=None):

        train_params = self.train_params
        mlflow_logging = self.mlflow_logging

        if mlflow_logging:
            try:
                import mlflow
            except ImportError:
                log.warning("Failed to import mlflow. MLflow logging is disabled.")
                mlflow_logging = False

        if mlflow_logging:

            if parse_version(ignite.__version__) >= parse_version("0.2.1"):
                from ignite.contrib.handlers.mlflow_logger import (
                    MLflowLogger,
                    OutputHandler,
                    global_step_from_engine,
                )
            else:
                from .ignite.contrib.handlers.mlflow_logger import (
                    MLflowLogger,
                    OutputHandler,
                    global_step_from_engine,
                )

        train_dataset_size_limit = train_params.get("train_dataset_size_limit")
        val_dataset_size_limit = train_params.get("val_dataset_size_limit")
        if train_dataset_size_limit:
            train_dataset = PartialDataset(train_dataset, train_dataset_size_limit)
            log.info("train dataset size is set to {}".format(len(train_dataset)))

        if val_dataset_size_limit:
            val_dataset = PartialDataset(val_dataset, val_dataset_size_limit)
            log.info("val dataset size is set to {}".format(len(val_dataset)))

        train_data_loader_params = train_params.get("train_data_loader_params", dict())
        val_data_loader_params = train_params.get("val_data_loader_params", dict())
        epochs = train_params.get("epochs")
        progress_update = train_params.get("progress_update")

        optimizer = train_params.get("optimizer")
        assert optimizer
        optimizer_params = train_params.get("optimizer_params", dict())
        scheduler = train_params.get("scheduler")
        scheduler_params = train_params.get("scheduler_params", dict())
        loss_fn = train_params.get("loss_fn")
        assert loss_fn
        evaluation_metrics = train_params.get("evaluation_metrics")

        evaluate_train_data = train_params.get("evaluate_train_data")
        evaluate_val_data = train_params.get("evaluate_val_data")

        early_stopping_params = train_params.get("early_stopping_params")
        time_limit = train_params.get("time_limit")
        model_checkpoint_params = train_params.get("model_checkpoint_params")
        seed = train_params.get("seed")
        cudnn_deterministic = train_params.get("cudnn_deterministic")
        cudnn_benchmark = train_params.get("cudnn_benchmark")

        if seed:
            torch.manual_seed(seed)
            np.random.seed(seed)
        if cudnn_deterministic:
            torch.backends.cudnn.deterministic = cudnn_deterministic
        if cudnn_benchmark:
            torch.backends.cudnn.benchmark = cudnn_benchmark

        optimizer_ = optimizer(model.parameters(), **optimizer_params)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        trainer = create_supervised_trainer(
            model, optimizer_, loss_fn=loss_fn, device=device
        )

        train_data_loader_params.setdefault("shuffle", True)
        train_data_loader_params.setdefault("drop_last", True)
        train_data_loader_params["batch_size"] = _clip_batch_size(
            train_data_loader_params.get("batch_size", 1), train_dataset, "train"
        )
        train_loader = DataLoader(train_dataset, **train_data_loader_params)

        RunningAverage(output_transform=lambda x: x, alpha=0.98).attach(
            trainer, "ema_loss"
        )

        RunningAverage(output_transform=lambda x: x, alpha=2 ** (-1022)).attach(
            trainer, "batch_loss"
        )

        if scheduler:

            class ParamSchedulerSavingAsMetric(
                ParamSchedulerSavingAsMetricMixIn, scheduler
            ):
                pass

            cycle_epochs = scheduler_params.pop("cycle_epochs", 1)
            scheduler_params.setdefault(
                "cycle_size", int(cycle_epochs * len(train_loader))
            )
            scheduler_params.setdefault("param_name", "lr")
            scheduler_ = ParamSchedulerSavingAsMetric(optimizer_, **scheduler_params)
            trainer.add_event_handler(Events.ITERATION_STARTED, scheduler_)

        if evaluate_train_data:
            evaluator_train = create_supervised_evaluator(
                model, metrics=evaluation_metrics, device=device
            )

        if evaluate_val_data:
            val_data_loader_params["batch_size"] = _clip_batch_size(
                val_data_loader_params.get("batch_size", 1), val_dataset, "val"
            )
            val_loader = DataLoader(val_dataset, **val_data_loader_params)
            evaluator_val = create_supervised_evaluator(
                model, metrics=evaluation_metrics, device=device
            )

        if model_checkpoint_params:
            assert isinstance(model_checkpoint_params, dict)
            minimize = model_checkpoint_params.pop("minimize", True)
            save_interval = model_checkpoint_params.get("save_interval", None)
            if not save_interval:
                model_checkpoint_params.setdefault(
                    "score_function", get_score_function("ema_loss", minimize=minimize)
                )
            model_checkpoint_params.setdefault("score_name", "ema_loss")
            mc = FlexibleModelCheckpoint(**model_checkpoint_params)
            trainer.add_event_handler(Events.EPOCH_COMPLETED, mc, {"model": model})

        if early_stopping_params:
            assert isinstance(early_stopping_params, dict)
            metric = early_stopping_params.pop("metric", None)
            assert (metric is None) or (metric in evaluation_metrics)
            minimize = early_stopping_params.pop("minimize", False)
            if metric:
                assert (
                    "score_function" not in early_stopping_params
                ), "Remove either 'metric' or 'score_function' from early_stopping_params: {}".format(
                    early_stopping_params
                )
                early_stopping_params["score_function"] = get_score_function(
                    metric, minimize=minimize
                )

            es = EarlyStopping(trainer=trainer, **early_stopping_params)
            if evaluate_val_data:
                evaluator_val.add_event_handler(Events.COMPLETED, es)
            elif evaluate_train_data:
                evaluator_train.add_event_handler(Events.COMPLETED, es)
            elif early_stopping_params:
                log.warning(
                    "Early Stopping is disabled because neither "
                    "evaluate_val_data nor evaluate_train_data is set True."
                )

        if time_limit:
            assert isinstance(time_limit, (int, float))
            tl = TimeLimit(limit_sec=time_limit)
            trainer.add_event_handler(Events.ITERATION_COMPLETED, tl)

        pbar = None
        if progress_update:
            if not isinstance(progress_update, dict):
                progress_update = dict()
            progress_update.setdefault("persist", True)
            progress_update.setdefault("desc", "")
            pbar = ProgressBar(**progress_update)
            pbar.attach(trainer, ["ema_loss"])

        else:

            def log_train_metrics(engine):
                log.info(
                    "[Epoch: {} | {}]".format(engine.state.epoch, engine.state.metrics)
                )

            trainer.add_event_handler(Events.EPOCH_COMPLETED, log_train_metrics)

        if evaluate_train_data:

            def log_evaluation_train_data(engine):
                evaluator_train.run(train_loader)
                train_report = _get_report_str(engine, evaluator_train, "Train Data")
                if pbar:
                    pbar.log_message(train_report)
                else:
                    log.info(train_report)

            eval_train_event = (
                Events[evaluate_train_data]
                if isinstance(evaluate_train_data, str)
                else Events.EPOCH_COMPLETED
            )
            trainer.add_event_handler(eval_train_event, log_evaluation_train_data)

        if evaluate_val_data:

            def log_evaluation_val_data(engine):
                evaluator_val.run(val_loader)
                val_report = _get_report_str(engine, evaluator_val, "Val Data")
                if pbar:
                    pbar.log_message(val_report)
                else:
                    log.info(val_report)

            eval_val_event = (
                Events[evaluate_val_data]
                if isinstance(evaluate_val_data, str)
                else Events.EPOCH_COMPLETED
            )
            trainer.add_event_handler(eval_val_event, log_evaluation_val_data)

        if mlflow_logging:
            mlflow_logger = MLflowLogger()

            logging_params = {
                "train_n_samples": len(train_dataset),
                "train_n_batches": len(train_loader),
                "optimizer": _name(optimizer),
                "loss_fn": _name(loss_fn),
                "pytorch_version": torch.__version__,
                "ignite_version": ignite.__version__,
            }
            logging_params.update(_loggable_dict(optimizer_params, "optimizer"))
            logging_params.update(_loggable_dict(train_data_loader_params, "train"))
            if scheduler:
                logging_params.update({"scheduler": _name(scheduler)})
                logging_params.update(_loggable_dict(scheduler_params, "scheduler"))

            if evaluate_val_data:
                logging_params.update(
                    {
                        "val_n_samples": len(val_dataset),
                        "val_n_batches": len(val_loader),
                    }
                )
                logging_params.update(_loggable_dict(val_data_loader_params, "val"))

            mlflow_logger.log_params(logging_params)

            batch_metric_names = ["batch_loss", "ema_loss"]
            if scheduler:
                batch_metric_names.append(scheduler_params.get("param_name"))

            mlflow_logger.attach(
                trainer,
                log_handler=OutputHandler(
                    tag="step",
                    metric_names=batch_metric_names,
                    global_step_transform=global_step_from_engine(trainer),
                ),
                event_name=Events.ITERATION_COMPLETED,
            )

            if evaluate_train_data:
                mlflow_logger.attach(
                    evaluator_train,
                    log_handler=OutputHandler(
                        tag="train",
                        metric_names=list(evaluation_metrics.keys()),
                        global_step_transform=global_step_from_engine(trainer),
                    ),
                    event_name=Events.COMPLETED,
                )
            if evaluate_val_data:
                mlflow_logger.attach(
                    evaluator_val,
                    log_handler=OutputHandler(
                        tag="val",
                        metric_names=list(evaluation_metrics.keys()),
                        global_step_transform=global_step_from_engine(trainer),
                    ),
                    event_name=Events.COMPLETED,
                )

        trainer.run(train_loader, max_epochs=epochs)

        try:
            if pbar and pbar.pbar:
                pbar.pbar.close()
        except Exception as e:
            log.error(e, exc_info=True)

        model = load_latest_model(model_checkpoint_params)(model)

        return model


def get_score_function(metric, minimize=False):
    def _score_function(engine):
        m = engine.state.metrics.get(metric)
        return -m if minimize else m

    return _score_function


def load_latest_model(model_checkpoint_params=None):
    if model_checkpoint_params and "model_checkpoint_params" in model_checkpoint_params:
        model_checkpoint_params = model_checkpoint_params.get("model_checkpoint_params")

    def _load_latest_model(model=None):
        if model_checkpoint_params:
            try:
                dirname = model_checkpoint_params.get("dirname")
                assert dirname
                dir_glob = Path(dirname).glob("*.pth")
                files = [str(p) for p in dir_glob if p.is_file()]
                if len(files) >= 1:
                    model_path = sorted(files)[-1]
                    log.info("Model path: {}".format(model_path))
                    loaded = torch.load(model_path)
                    save_as_state_dict = model_checkpoint_params.get(
                        "save_as_state_dict", True
                    )
                    if save_as_state_dict:
                        assert model
                        model.load_state_dict(loaded)
                    else:
                        model = loaded
                else:
                    log.warning("Model not found at: {}".format(dirname))
            except Exception as e:
                log.error(e, exc_info=True)
        return model

    return _load_latest_model


""" https://github.com/pytorch/ignite/blob/v0.2.1/ignite/handlers/checkpoint.py """


class FlexibleModelCheckpoint(ModelCheckpoint):
    def __init__(
        self,
        dirname,
        filename_prefix,
        offset_hours=0,
        filename_format=None,
        suffix_format=None,
        *args,
        **kwargs
    ):
        if "%" in filename_prefix:
            filename_prefix = get_timestamp(
                fmt=filename_prefix, offset_hours=offset_hours
            )
        super().__init__(dirname, filename_prefix, *args, **kwargs)
        if not callable(filename_format):
            if isinstance(filename_format, str):
                format_str = filename_format
            else:
                format_str = "{}_{}_{:06d}{}.pth"

            def filename_format(filename_prefix, name, step_number, suffix):
                return format_str.format(filename_prefix, name, step_number, suffix)

        self._filename_format = filename_format

        if not callable(suffix_format):
            if isinstance(suffix_format, str):
                suffix_str = suffix_format
            else:
                suffix_str = "_{}_{:.7}"

            def suffix_format(score_name, abs_priority):
                return suffix_str.format(score_name, abs_priority)

        self._suffix_format = suffix_format

    def __call__(self, engine, to_save):
        if len(to_save) == 0:
            raise RuntimeError("No objects to checkpoint found.")

        self._iteration += 1

        if self._score_function is not None:
            priority = self._score_function(engine)

        else:
            priority = self._iteration
            if (self._iteration % self._save_interval) != 0:
                return

        if (len(self._saved) < self._n_saved) or (self._saved[0][0] < priority):
            saved_objs = []

            suffix = ""
            if self._score_name is not None:
                suffix = self._suffix_format(self._score_name, abs(priority))

            for name, obj in to_save.items():
                fname = self._filename_format(
                    self._fname_prefix, name, self._iteration, suffix
                )
                path = os.path.join(self._dirname, fname)

                self._save(obj=obj, path=path)
                saved_objs.append(path)

            self._saved.append((priority, saved_objs))
            self._saved.sort(key=lambda item: item[0])

        if len(self._saved) > self._n_saved:
            _, paths = self._saved.pop(0)
            for p in paths:
                os.remove(p)


""" """


def get_timestamp(fmt="%Y-%m-%dT%H:%M:%S", offset_hours=0):
    return (datetime.now() + timedelta(hours=offset_hours)).strftime(fmt)


def _name(obj):
    return getattr(obj, "__name__", None) or getattr(obj.__class__, "__name__", "_")


def _clip_batch_size(batch_size, dataset, tag=""):
    dataset_size = len(dataset)
    if batch_size > dataset_size:
        log.warning(
            "[{}] batch size ({}) is clipped to dataset size ({})".format(
                tag, batch_size, dataset_size
            )
        )
        return dataset_size
    else:
        return batch_size


def _get_report_str(engine, evaluator, tag=""):
    report_str = "[Epoch: {} | {} | Metrics: {}]".format(
        engine.state.epoch, tag, evaluator.state.metrics
    )
    return report_str


def _loggable_dict(d, prefix=None):
    return {
        ("{}_{}".format(prefix, k) if prefix else k): (
            "{}".format(v) if isinstance(v, (tuple, list, dict, set)) else v
        )
        for k, v in d.items()
    }


class TimeLimit:
    def __init__(self, limit_sec=3600):
        self.limit_sec = limit_sec
        self.start_time = time.time()

    def __call__(self, engine):
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.limit_sec:
            log.warning(
                "Reached the time limit: {} sec. Stop training".format(self.limit_sec)
            )
            engine.terminate()


class ParamSchedulerSavingAsMetricMixIn:
    """ Base code:
     https://github.com/pytorch/ignite/blob/v0.2.1/ignite/contrib/handlers/param_scheduler.py#L49
     https://github.com/pytorch/ignite/blob/v0.2.1/ignite/contrib/handlers/param_scheduler.py#L163
    """

    def __call__(self, engine, name=None):

        if self.event_index != 0 and self.event_index % self.cycle_size == 0:
            self.event_index = 0
            self.cycle_size *= self.cycle_mult
            self.cycle += 1
            self.start_value *= self.start_value_mult
            self.end_value *= self.end_value_mult

        value = self.get_param()

        for param_group in self.optimizer_param_groups:
            param_group[self.param_name] = value

        if name is None:
            name = self.param_name

        if self.save_history:
            if not hasattr(engine.state, "param_history"):
                setattr(engine.state, "param_history", {})
            engine.state.param_history.setdefault(name, [])
            values = [pg[self.param_name] for pg in self.optimizer_param_groups]
            engine.state.param_history[name].append(values)

        self.event_index += 1

        if not hasattr(engine.state, "metrics"):
            setattr(engine.state, "metrics", {})
        engine.state.metrics[self.param_name] = value  # Save as a metric


class PartialDataset:
    def __init__(self, dataset, size):
        size = int(size)
        assert hasattr(dataset, "__getitem__")
        assert hasattr(dataset, "__len__")
        assert dataset.__len__() >= size
        self.dataset = dataset
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        return self.dataset[item]


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


def element_wise_sum(tt_list):
    return torch.sum(torch.stack(tt_list), dim=0)


class ModuleSum(ModuleListMerge):
    def forward(self, input):
        tt_list = super().forward(input)
        assert len(set([tuple(list(tt.size())) for tt in tt_list])) == 1, (
            "Sizes of tensors must match. "
            "\n{}\n got tensor sizes: \n{}\n".format(
                self, [tt.size() for tt in tt_list]
            )
        )
        return element_wise_sum(tt_list)


def element_wise_average(tt_list):
    return torch.mean(torch.stack(tt_list), dim=0)


class ModuleAvg(ModuleListMerge):
    def forward(self, input):
        tt_list = super().forward(input)
        assert len(set([tuple(list(tt.size())) for tt in tt_list])) == 1, (
            "Sizes of tensors must match. "
            "\n{}\n got tensor sizes: \n{}\n".format(
                self, [tt.size() for tt in tt_list]
            )
        )
        return element_wise_average(tt_list)


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
    kernel_size=(1, 1),
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


class ModuleConvWrap(torch.nn.Sequential):
    core = None

    def __init__(self, activation=None, *args, **kwargs):
        args, kwargs = setup_conv_params(*args, **kwargs)
        module = self.core(*args, **kwargs)
        if activation:
            super().__init__(module, activation)
        else:
            super().__init__(module)


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


class TensorIdentity(TensorSkip):
    pass


class TensorForward(torch.nn.Module):
    def __init__(self, func=None):
        func = func or (lambda x: x)
        assert callable(func)
        self._func = func

    def forward(self, input):
        return self._func(input)


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


def nl_loss(input, *args, **kwargs):
    return torch.nn.functional.nll_loss(input.log(), *args, **kwargs)


class NLLoss(torch.nn.NLLLoss):
    """ The negative likelihood loss.
    To compute Cross Entropy Loss, there are 3 options.
        NLLoss with torch.nn.Softmax
        torch.nn.NLLLoss with torch.nn.LogSoftmax
        torch.nn.CrossEntropyLoss
    """

    def forward(self, input, target):
        return super().forward(input.log(), target)


class TransformCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, d):
        for t in self.transforms:
            d = t(d)
        return d


_to_channel_last_dict = {3: (-2, -1, -3), 4: (0, -2, -1, -3)}


def to_channel_last_arr(a):
    if a.ndim in {3, 4}:
        return np.transpose(a, axes=_to_channel_last_dict.get(a.ndim))
    else:
        return a


_to_channel_first_dict = {3: (-1, -3, -2), 4: (0, -1, -3, -2)}


def to_channel_first_arr(a):
    if a.ndim in {3, 4}:
        return np.transpose(a, axes=_to_channel_first_dict.get(a.ndim))
    else:
        return a
