"""Helper methods for using Keras, often using keract

Yields:
    [type]: [description]
"""
from typing import Tuple

import keract
import numpy as np
import keras.backend as K
from tqdm.auto import tqdm

from . import dbhandler
from . import neural_networks as nn
from .frame_extractor import FrameExtractor
from .image_preprocessing import process_image


def add_layer_activations_to_database(
    network: nn.Network, database_file: str, frames: FrameExtractor
):
    """Populates the database with with he layer from all images described by the
    FrameExtractor.

    The specific neural network and layer to use is described by the Network class.

    Args:
        network (nn.Network): The Network class describing which network and layer to use
        database_file (str): Where to find the database file
        frames (FrameExtractor): The thumnbnail generator, which includes a list of all frames

    Returns:
        Tuple[dict, dict]: [description]
    """
    model = network.used_model
    used_model_layer = network.default_layer
    target_size = network.target_size

    if network.stop_at_layer is None:
        get_layer_outputs = K.function(
            [model.layers[0].input], model.layers[used_model_layer].output
        )
    else:
        get_layer_outputs = K.function(
            [model.layers[0].input], model.get_layer(network.stop_at_layer).output
        )

    with dbhandler.VRDDatabase(database_file) as dbc:
        for img in tqdm(frames.all_images):
            db_activation = dbc.get_layer_data(img)
            if db_activation is None:
                layer_data = get_layer_outputs(
                    process_image(img, target_size, trim=True)
                )
                value = np.reshape(layer_data, (1, np.product(layer_data.shape)))
                dbc.add_layer_data(img, value)

    # return results_dict, results_dict_full


# as per https://stackoverflow.com/a/8290508
def batch(iterable, batch_size=1):
    """Split a long list into separate batches according to batch_size"""
    iter_length = len(iterable)
    for ndx in range(0, iter_length, batch_size):
        yield iterable[ndx : min(ndx + batch_size, iter_length)]


def display_activation_for_image(filename: str, network: nn.Network):
    """Uses keract to display activation for a given image!

    Args:
        filename (str): The image
    """
    activations = keract.get_activations(
        network.used_model,
        process_image(filename, network.target_size, trim=True),
        auto_compile=True,
    )
    keract.display_activations(
        activations,
        cmap=None,
        save=False,
        directory=".",
        data_format="channels_last",
        fig_size=(24, 24),
        reshape_1d_layers=False,
    )


def get_layer_info(network: nn.Network):
    """Creates a Pandas DataFrame-compliant list of dicts
    with information of each layer including index, name, output shape
    and total number of elements.

    This can greatly assist in selecting a reasonable layer for using the VRD.

    Args:
        network (nn.Network): The network to use as a base

    Returns:
        list: A list of dicts, which can then be used by pandas (or justprinted)
    """

    model = network.used_model


    layer_info = []

    def _get_output_size(output_shape):
        if shape is None:
            return 0
        return np.prod([x for x in output_shape if x is not None]) 

    for idx, layer in enumerate(model.layers):
        shape = layer.output_shape
        if type(shape) is list:
            if len(shape) == 0:
                shape = None
            else:
                shape = shape[0]
        # pprint(shape)
        layer_info.append(
            {'Index':idx,
            'Name':layer.name,
            'Output size':shape,
            'Output element count':_get_output_size(shape)
            })
    
    return layer_info
