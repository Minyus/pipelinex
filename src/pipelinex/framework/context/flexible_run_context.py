import logging.config
import logging
from typing import Any, Dict, Iterable, Optional, Union  # NOQA
from warnings import warn
import kedro
from kedro.versioning import Journal
from .context import KedroContext, KedroContextError
from kedro.runner import AbstractRunner, SequentialRunner
import kedro.runner

from .save_pipeline_json_context import SavePipelineJsonContext

log = logging.getLogger(__name__)


class OnlyMissingOptionContext(KedroContext):
    """Users can override the remaining methods from the parent class here, or create new ones
    (e.g. as required by plugins)

    """

    def run(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        tags: Iterable[str] = None,
        runner: AbstractRunner = None,
        node_names: Iterable[str] = None,
        from_nodes: Iterable[str] = None,
        to_nodes: Iterable[str] = None,
        from_inputs: Iterable[str] = None,
        load_versions: Dict[str, str] = None,
        pipeline_name: str = None,
        only_missing: bool = False,
    ) -> Dict[str, Any]:
        """Runs the pipeline with a specified runner.

        Args:
            tags: An optional list of node tags which should be used to
                filter the nodes of the ``Pipeline``. If specified, only the nodes
                containing *any* of these tags will be run.
            runner: An optional parameter specifying the runner that you want to run
                the pipeline with.
            node_names: An optional list of node names which should be used to
                filter the nodes of the ``Pipeline``. If specified, only the nodes
                with these names will be run.
            from_nodes: An optional list of node names which should be used as a
                starting point of the new ``Pipeline``.
            to_nodes: An optional list of node names which should be used as an
                end point of the new ``Pipeline``.
            from_inputs: An optional list of input datasets which should be used as a
                starting point of the new ``Pipeline``.
            load_versions: An optional flag to specify a particular dataset version timestamp
                to load.
            pipeline_name: Name of the ``Pipeline`` to execute.
                Defaults to "__default__".
            only_missing: An option to run only missing nodes.
        Raises:
            KedroContextError: If the resulting ``Pipeline`` is empty
                or incorrect tags are provided.
            Exception: Any uncaught exception will be re-raised
                after being passed to``on_pipeline_error``.
        Returns:
            Any node outputs that cannot be processed by the ``DataCatalog``.
            These are returned in a dictionary, where the keys are defined
            by the node outputs.
        """
        # Report project name
        logging.info("** Kedro project %s", self.project_path.name)

        try:
            pipeline = self._get_pipeline(name=pipeline_name)
        except NotImplementedError:
            common_migration_message = (
                "`ProjectContext._get_pipeline(self, name)` method is expected. "
                "Please refer to the 'Modular Pipelines' section of the documentation."
            )
            if pipeline_name:
                raise KedroContextError(
                    "The project is not fully migrated to use multiple pipelines. "
                    + common_migration_message
                )

            warn(
                "You are using the deprecated pipeline construction mechanism. "
                + common_migration_message,
                DeprecationWarning,
            )
            pipeline = self.pipeline

        filtered_pipeline = self._filter_pipeline(
            pipeline=pipeline,
            tags=tags,
            from_nodes=from_nodes,
            to_nodes=to_nodes,
            node_names=node_names,
            from_inputs=from_inputs,
        )

        if hasattr(self, "_save_pipeline_json"):
            self._save_pipeline_json(filtered_pipeline)

        save_version = self._get_save_version()
        run_id = self.run_id or save_version

        record_data = {
            "run_id": run_id,
            "project_path": str(self.project_path),
            "env": self.env,
            "kedro_version": self.project_version,
            "tags": tags,
            "from_nodes": from_nodes,
            "to_nodes": to_nodes,
            "node_names": node_names,
            "from_inputs": from_inputs,
            "load_versions": load_versions,
            "pipeline_name": pipeline_name,
            "extra_params": self._extra_params,
        }
        journal = Journal(record_data)

        catalog = self._get_catalog(
            save_version=save_version, journal=journal, load_versions=load_versions
        )

        # Run the runner
        runner = runner or SequentialRunner()
        self._hook_manager.hook.before_pipeline_run(  # pylint: disable=no-member
            run_params=record_data, pipeline=filtered_pipeline, catalog=catalog
        )

        try:
            run_method = runner.run_only_missing if only_missing else runner.run
            if run_method.__code__.co_argcount >= 4:
                # if kedro.__version__ >= "0.16.0"
                run_result = run_method(filtered_pipeline, catalog, run_id)
            else:
                # if kedro.__version__ <= "0.15.9"
                run_result = run_method(filtered_pipeline, catalog)

        except Exception as error:
            self._hook_manager.hook.on_pipeline_error(  # pylint: disable=no-member
                error=error,
                run_params=record_data,
                pipeline=filtered_pipeline,
                catalog=catalog,
            )
            raise error

        self._hook_manager.hook.after_pipeline_run(  # pylint: disable=no-member
            run_params=record_data,
            run_result=run_result,
            pipeline=filtered_pipeline,
            catalog=catalog,
        )
        return run_result


class StringRunnerOptionContext(KedroContext):
    """Allow to specify runner by string."""

    def run(
        self,
        *args,  # type: Any
        runner=None,  # type: Union[AbstractRunner, str]
        **kwargs,  # type: Any
    ):
        # type: (...) -> Dict[str, Any]
        runner = runner or "SequentialRunner"
        if isinstance(runner, str):
            runner = getattr(kedro.runner, runner)()
        return super().run(*args, runner=runner, **kwargs)


class FlexibleRunContext(
    SavePipelineJsonContext, StringRunnerOptionContext, OnlyMissingOptionContext
):
    pass
