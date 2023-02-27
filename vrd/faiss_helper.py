"""A file containting some helper files for the faiss library

"""
# pylint: disable=no-value-for-parameter
import os
import pickle

from operator import itemgetter
from typing import TYPE_CHECKING

import faiss
import numpy as np
from tqdm.auto import tqdm

from . import dbhandler, frame_extractor, neighbours, neural_networks

if TYPE_CHECKING:
    from .neighbours import Neighbours


def get_faiss_index(
    database_file: str,
    network: neural_networks.Network,
    frames: frame_extractor.FrameExtractor,
):
    """Creates a faiss index from all the frames in the specified FrameExtractor.
    This requires the database to be already populated with valid layer data

    Args:
        database_file (str): The database containing the layer data
        network (neural_networks.Network): The network used to create the data
        frames (frame_extractor.FrameExtractor): The frame extractor describing the included files

    Returns:
        A faiss index, filled with indexes
    """

    def get_layer_value_count(x):
        return np.asscalar(
            np.prod([i for i in x.output_shape if i])
        )  # TODO: Depricated asscalar

    layer_size = get_layer_value_count(network.used_model.layers[network.default_layer])

    faiss_index = faiss.IndexFlatL2(layer_size)

    with dbhandler.VRDDatabase(database_file) as dbc:
        for img in tqdm(frames.all_images):
            # Maybe more stable if we do batches instead?
            db_activation = dbc.get_layer_data(img)
            # Here we would add any (optional) preprocessing steps,
            # e.g. centering or scaling of some sort
            faiss_index.add(db_activation)
    return faiss_index


def get_faiss_index_fpq(
    database_file: str,
    network: neural_networks.Network,
    frames: frame_extractor.FrameExtractor,
    quantizer=None,
    concat=8,
    nlist=100,
):
    """Creates a faiss index from all the frames in the specified FrameExtractor.
    This requires the database to be already populated with valid layer data

    Args:
        database_file (str): The database containing the layer data
        network (neural_networks.Network): The network used to create the data
        frames (frame_extractor.FrameExtractor): The frame extractor describing the included files
        quantizer


    Returns:
        A faiss index, filled with indexes
    """

    def get_layer_value_count(x):
        return np.asscalar(
            np.prod([i for i in x.output_shape if i])
        )  # TODO: Depricated asscalar

    layer_size = get_layer_value_count(network.used_model.layers[network.default_layer])
    # m = 8                             # number of subquantizers
    quantizer = faiss.IndexFlatL2(layer_size)
    index = faiss.IndexIVFPQ(quantizer, layer_size, nlist, concat, 8)

    # First train on a subset...
    number_to_train = 40000
    train_indexes = np.linspace(
        0, len(frames.all_images) - 1, number_to_train, dtype=int
    )

    with dbhandler.VRDDatabase(database_file) as dbc:
        activations = []
        for img in tqdm(itemgetter(*train_indexes)(frames.all_images)):
            activations.append(dbc.get_layer_data(img))
        index.train(np.concatenate(activations))
    # add remaining
    print("Finished training")
    with dbhandler.VRDDatabase(database_file) as dbc:
        activations = []
        for img in tqdm(frames.all_images):
            # Maybe more stable if we do batches instead?
            activations.append(dbc.get_layer_data(img))

            if len(activations) > 20000:
                concat = np.concatenate(activations)
                # Here we would add any (optional) preprocessing steps, e.g.
                # centering or scaling of some sort

                index.add(concat)
                activations = []
            # faiss_index.add(db_activation)
        if len(activations) > 0:
            concat = np.concatenate(activations)
            # Here we would add any (optional) preprocessing steps, e.g.
            # centering or scaling of some sort
            index.train(concat)
            index.add(concat)
            activations = []
    return index


# as per https://stackoverflow.com/a/8290508
def batch(iterable, batch_size=1):
    """Helper function to run a large number of iteratebles in batches"""
    length = len(iterable)
    for ndx in range(0, length, batch_size):
        yield iterable[ndx : min(ndx + batch_size, length)]


def _correct_indexes(distance_list):
    """Ensures that indexes are correctly sorted.

    This implies that the "source" frame is always at index 0.

    This can be incorrect in case there are duplicates in the faiss database.

    Args:
        distance_list ([type]): The distance list

    Returns:
        A distance list with corrected indexes
    """
    for expected_idx in range(0, len(distance_list)):
        d, i = distance_list[expected_idx]
        if i[0] != expected_idx:  # Couldn't find correct index in list
            correct_loc = list(np.nonzero(i == expected_idx)[0])
            if len(correct_loc) == 0:  # If not found
                d = np.insert(d, 0, 0)
                i = np.insert(i, 0, expected_idx)
            else:  # Correct index was found.
                # Swap locations. We assume distance is the same.
                i[correct_loc] = i[0]
                i[0] = expected_idx
            distance_list[expected_idx] = (d, i)
    return distance_list


def calculate_distance_list(
    frames: frame_extractor.FrameExtractor,
    database_file: str,
    faiss_index,
    neighbour_num=100,
    batch_size=1000,
    batch_directory=None,
):
    """Find the neighbour_num closest matches to each"""
    current_batch_start = 0
    all_images = frames.all_images
    # gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
    # print('Opening DB....')
    with dbhandler.VRDDatabase(database_file) as dbc:
        distance_list = []
        for idx, curr_batch in tqdm(
            enumerate(batch(range(len(all_images)), batch_size)),
            total=np.ceil(len(all_images) / batch_size),
        ):
            if batch_directory is not None and os.path.exists(
                f"{batch_directory}/batch_{idx}.pickle"
            ):
                with open(f"{batch_directory}/batch_{idx}.pickle", "rb") as file:
                    print(f"Batch {idx}, found pickled batch. Loading...")
                    d, i, curr_batch_loaded = pickle.load(file)
                    if curr_batch != curr_batch_loaded:
                        print(
                            "Saved batch ranges do not match!"
                        )  # Consider if we should actually do something...
                        return None

            else:
                profiles = np.array(
                    [dbc.get_layer_data(all_images[x]).flatten() for x in curr_batch]
                )
                d, i = faiss_index.search(profiles, neighbour_num)
                if batch_directory is not None:
                    with open(f"{batch_directory}/batch_{idx}.pickle", "wb") as file:
                        pickle.dump([d, i, curr_batch], file)
            for combined in zip(d, i):
                distance_list.append(combined)

            current_batch_start += len(i)

    d_list = _correct_indexes(distance_list)

    # Sanity check for uniqueness
    actual_unique = np.unique([x[1][0] for x in d_list])
    if len(actual_unique) != len(distance_list):
        print(
            f"Warning: Incorrect number of unique indexes in distance list.\nExpected: {len(distance_list)}, got {len(actual_unique)}"
        )
    return neighbours.Neighbours(frames, distance_list)


def calculate_distance_list_external(
    external_database_file: str,
    reference_faiss_index: faiss.Index,
    external_frames: frame_extractor.FrameExtractor,
    neighbour_num=100,
    batch_size=1000,
    batch_directory=None,
):
    """Compare one VRD project to another, i.e. it's not an all-against-all comparison."""
    current_batch_start = 0
    external_images = external_frames.all_images
    images_to_predict = external_images

    # gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
    # print('Opening DB....')
    with dbhandler.VRDDatabase(external_database_file) as dbc:
        distance_list = []
        for idx, curr_batch in tqdm(
            enumerate(batch(range(len(images_to_predict)), batch_size)),
            total=np.ceil(len(images_to_predict) / batch_size),
        ):
            if batch_directory is not None and os.path.exists(
                f"{batch_directory}/batch_{idx}.pickle"
            ):
                with open(f"{batch_directory}/batch_{idx}.pickle", "rb") as file:
                    print(f"Batch {idx}, found pickled batch. Loading...")
                    d, i, curr_batch_loaded = pickle.load(file)
                    if curr_batch != curr_batch_loaded:
                        print(
                            "Saved batch ranges do not match!"
                        )  # Consider if we should actually do something...
                        return None

            else:
                # print(f'attempting to open {images_to_predict[x]}')
                flattened_fingerprints = []
                for x in curr_batch:
                    saved_data = dbc.get_layer_data(images_to_predict[x])
                    if saved_data is None:
                        print(
                            f"Could not find saved fingerprint for image {images_to_predict[x]}, aborting..."
                        )
                        return None
                    flattened_fingerprints.append(saved_data.flatten())

                profiles = np.array(flattened_fingerprints)
                d, i = reference_faiss_index.search(profiles, neighbour_num)
                if batch_directory is not None:
                    with open(f"{batch_directory}/batch_{idx}.pickle", "wb") as file:
                        pickle.dump([d, i, curr_batch], file)
            for combined in zip(d, i):
                distance_list.append(combined)

            current_batch_start += len(i)
    return distance_list
