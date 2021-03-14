from kedro.pipeline import Pipeline

from .context import KedroContext


class SavePipelineJsonContext(KedroContext):
    _pipeline_json_text_dataset = None

    def _save_pipeline_json(self, pipeline: Pipeline):
        if self._pipeline_json_text_dataset is not None:
            pipeline_json_str = pipeline.to_json()
            self._pipeline_json_text_dataset.save(pipeline_json_str)
