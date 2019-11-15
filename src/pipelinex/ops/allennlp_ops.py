def allennlp_reader_to_dict(**kwargs):
    def _reader_to_dict(*argsignore, **kwargsignore):
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

    return _reader_to_dict
