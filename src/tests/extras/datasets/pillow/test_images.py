import tempfile

import numpy as np

from pipelinex import ImagesLocalDataSet


def test_save_and_load():

    with tempfile.TemporaryDirectory() as dir:

        ds = ImagesLocalDataSet(
            path=dir + "/foobar_images", save_args={"suffix": ".jpg"}
        )

        image_dict = {
            "foo_image": 255 * np.ones((480, 640, 3), dtype=np.uint8),
            "bar_image": np.zeros((100, 200, 3), dtype=np.uint8),
        }

        ds._save(image_dict)
        loaded_image_dict = ds._load()

        for name in image_dict.keys():
            assert (image_dict[name] == loaded_image_dict[name]).all()


if __name__ == "__main__":
    test_save_and_load()
