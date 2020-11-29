from logging import getLogger


log = getLogger(__name__)

try:
    from kedro.framework.context import *  # NOQA

    from kedro.framework.hooks import get_hook_manager

except (ImportError, ModuleNotFoundError):
    log.warning(
        "Failed to import context and hooks from kedro.framework."
        "Hooks will not be registered. Recommended to update Kedro to >= 0.16"
    )
    # if kedro.__version__ <= "0.15.9"

    from kedro.context import *  # NOQA

    def get_hook_manager():
        class Hook:
            def after_catalog_created(self, *args, **kwargs):
                pass

            def before_node_run(self, *args, **kwargs):
                pass

            def after_node_run(self, *args, **kwargs):
                pass

            def on_node_error(self, *args, **kwargs):
                pass

            def before_pipeline_run(self, *args, **kwargs):
                pass

            def after_pipeline_run(self, *args, **kwargs):
                pass

            def on_pipeline_error(self, *args, **kwargs):
                pass

        hook = Hook()

        class HookManager:
            def __init__(self):
                self.hook = hook

            def is_registered(self, *args, **kwargs):
                return False

            def register(self, *args, **kwargs):
                pass

        return HookManager()
