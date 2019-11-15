from sklearn.model_selection import train_test_split


def df_train_test_split(**kwargs):
    def _df_train_test_split(df, *argsignore, **kwargsignore):
        return train_test_split(df, **kwargs)

    return _df_train_test_split
