class AllennlpReaderToDict:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, *args_ignore, **kwargs_ignore):
        kwargs = self.kwargs

        reader = kwargs.get("reader")
        file_path = kwargs.get("file_path")
        n_samples = kwargs.get("n_samples")
        instances = reader._read(file_path)
        n_samples = n_samples or len(instances)
        d = dict()
        i = 0
        for instance in instances:
            if n_samples and i >= n_samples:
                break
            d[i] = instance.fields
            i += 1
        return d
