"""Contains the VRD Project class
"""

import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from faiss import read_index, swigfaiss_avx2, write_index

from . import faiss_helper
from . import keras_layer_helper as klh
from .frame_extractor import FrameExtractor
from .neighbours import Neighbours
from .neural_networks import Network, NeuralNetworks, get_network


@dataclass
class VRDProject:
    """A file containing all necessary information of a single VRD project.

    This includes frame extractor, frame fingerprint database, 
    neighbour calculations, relevant directories, settings, and so on.

    Returns:
        VRDProject: A VRD project object
    """

    name: str
    project_base_path: str
    video_path: str
    network: NeuralNetworks
    additional_video_extensions: list = None
    neighbours_considered: int = 250
    override_network_default_layer: int = -1
    stop_at_layer: str = None
    frame_extractor: FrameExtractor = None
    faiss_index: swigfaiss_avx2.Index = None
    neighbours: Neighbours = None
    _initialized_model: Network = None

    @property
    def video_extensions(self):
        """The video extensions to look for in this project

        Returns:
            list: A list of strings with extensions
        """
        default_extensions = ["avi", "mkv"]
        if self.additional_video_extensions is not None:
            return default_extensions + self.additional_video_extensions
        return default_extensions

    @property
    def project_path(self) -> str:
        """The path of the project, i.e. where files are saved.
            This is a combination of the "base path" (where all projects are saved)
            and the project name
        """
        return str(Path.joinpath(Path(self.project_base_path), Path(self.name)))

    @property
    def faiss_index_location(self) -> str:
        """The location of the saved faiss index file, inside the project path

        Returns:
            str: the path as a string
        """
        return str(Path.joinpath(Path(self.project_path), Path("faiss_index")))

    @property
    def frame_path(self) -> str:
        """The path to the project-related frames

        Returns:
            str: The path (as a string) to the frame directory
        """
        frame_path = Path.joinpath(Path(self.project_path), Path("frames/"))
        return str(frame_path)

    @property
    def neighbour_batch_path(self) -> str:
        """The path for the neighbour batch files.

        These files are used to save paratial neighbour results in case of a crash

        Returns:
            str: The directory path where neighbour batch files are saved
        """
        neighbour_batch_path = Path.joinpath(
            Path(self.project_path), Path("neighbour_batch/")
        )
        neighbour_batch_path.mkdir(exist_ok=True)
        return str(neighbour_batch_path)

    @property
    def database_file(self) -> str:
        """The location of the database file

        Returns:
            str: _description_
        """
        return str(Path.joinpath(Path(self.project_path), Path("database_file")))

    def ensure_exists(self, verbose=False):
        """Verify that the most necessary paths actually exist.

        Args:
            verbose (bool, optional): Show extra print outs. Defaults to False.
        """
        assert Path(
            self.video_path
        ).exists(), f"Video path ({self.video_path}) does not exist"
        assert Path(
            self.project_base_path
        ).exists(), f"Project base path ({self.project_base_path}) does not exist"
        assert Path(
            self.project_path
        ).exists(), f"Project path ({self.project_path}) does not exist"
        assert self.network is not None
        assert Path(
            self.frame_path
        ).exists(), f"Frame path ({self.frame_path}) does not exist"
        assert Path(
            self.database_file
        ).exists(), f"Database file ({self.database_file}) does not exist"
        if verbose:
            print("All directories exist.")

    def populate_fingerprint_database(self, force_recreate=False):
        """Populates the fingerprint database with fingerprints for all frames
        in the current frame_extractor. 

        If they already exist, they are not recalculated. Because of this, in cases
        where they need to be recalculated the force_recreate tag should be set to true,
        which deletes the old database first. This is necessary when changing layer or 
        changing neural network, as the results will otherwise not be updated.

        Args:
            force_recreate (bool, optional): _description_. Defaults to False.
        """
        if force_recreate:
            os.remove(self.database_file)
        model = self.get_model_instance()
        stop_layer = model.stop_at_layer
        if stop_layer is not None:
            print(f'Layer specified ({stop_layer}), model index ({model.default_layer}) will be ignored.')

        klh.add_layer_activations_to_database(
            model, self.database_file, self.frame_extractor
        )

    def initialize_frame_extractor(self, force_recreate=False):
        """Initialize the frame extractor, using the settings from the project.

        This will create all frames if they are not already created.

        Args:
            force_recreate (bool, optional): Create a new frame extractor, regardless if one
            was already loaded
        """
        if not force_recreate and self.frame_extractor is not None:
            return
        self.frame_extractor = FrameExtractor(
            self.video_path,
            self.frame_path,
            additional_video_extensions=self.video_extensions,
        )

    def get_model_instance(self, force_recreate=False) -> Network:
        """Get an instantiated version of the set model type.


        Args:
            force_recreate (bool, optional): Recalculates the model even if it exists,
                for example if the model type or layer has changed. Defaults to False.

       
        Returns:
            Network: An instance of the specified model
        
        """
        if self._initialized_model is not None and not force_recreate:
            return self._initialized_model
        neural_network = get_network(self.network)
        if self.override_network_default_layer > 0:
            neural_network.default_layer = self.override_network_default_layer
        neural_network.stop_at_layer = self.stop_at_layer

        self._initialized_model = neural_network
        return self._initialized_model

    def initialize_faiss_index(self, force_recreate=False):
        if not force_recreate and self.faiss_index is not None:
            return
        # Check if exists
        index_location = self.faiss_index_location
        if not force_recreate and Path(index_location).exists():
            print("Attempting to load faiss index...")
            try:
                self.faiss_index = read_index(index_location)
                print("Success.")
                return
            except:
                print("Failed to load existing faiss index!")
        # We create it...
        self.faiss_index = faiss_helper.get_faiss_index(
            self.database_file, self.get_model_instance(), self.frame_extractor
        )
        write_index(self.faiss_index, self.faiss_index_location)

    def initialize_neighbours(self, force_recreate=False):
        """Create the neighbour list, if it doesn't exist.

        Even if it does exist it can be recreated by using force_recreate = True

        Args:
            force_recreate (bool, optional): Force recreation of neighbours list, perhaps 
            if number of neighbours has changed
        """
        if not force_recreate and self.neighbours is not None:
            return
        if force_recreate:
            for pickle_file in glob(f"{self.neighbour_batch_path}/*.pickle"):
                os.remove(pickle_file)
        neighbours = faiss_helper.calculate_distance_list(
            frames=self.frame_extractor,
            database_file=self.database_file,
            faiss_index=self.faiss_index,
            neighbour_num=self.neighbours_considered,
            batch_directory=self.neighbour_batch_path,
        )
        self.neighbours = neighbours

    def show_most_reused_files(
        self, sequences, video_list=None, maximum_to_show=80, size=(1200, 600)
    ):
        """Creates a stacked bar plot showing which video file has been most reused,
        and which file reused them.
        
        Should this be here or in notebook helper?

        Args:
            sequences (list): A sequence list
            video_list (list, optional): An external list of videos to use 
                (generally fewer than otherwise). Defaults to None.
            maximum_to_show (int, optional): Maximum videos to show int the bar plot. 
                Defaults to 80.
            size (tuple, optional): The rendering size of the plot. Defaults to (1200, 600).

        Returns:
            figure: A figure which can be displayed, e.g. in a jupyter notebook
        """

        seq_matching = defaultdict(list)

        for start1, start2, dur in sequences:
            vid1, s1_time = self.frame_extractor.get_video_and_start_time_from_index(
                start1
            )
            vid2, s2_time = self.frame_extractor.get_video_and_start_time_from_index(
                start2
            )
            seq_matching[vid1].append((vid2, s1_time, s2_time, dur))
            seq_matching[vid2].append((vid1, s2_time, s1_time, dur))

        result_list = []
        if video_list is None:
            video_list = self.frame_extractor.video_list

        for vid in video_list:
            matches = dict(Counter([x[0] for x in seq_matching[vid]]))
            matches["Source video"] = vid
            result_list.append(matches)
        df = pd.DataFrame(result_list).set_index("Source video")
        df["sum"] = df.sum(axis=1).values
        df = df.sort_values(by="sum", ascending=False)

        df = df.drop(columns="sum")

        df2 = pd.DataFrame(df.stack())
        df2.columns = ["Reuse count"]
        df2 = df2.reset_index()
        df2 = df2.rename(columns={"level_1": "reference video"})
        videos_to_show = set(
            df.sum(axis=0).sort_values(ascending=False).head(maximum_to_show).index
        )
        df2 = df2[df2["reference video"].isin(videos_to_show)]
        width, height = size
        fig = px.bar(
            df2,
            x="reference video",
            y="Reuse count",
            color="Source video",
            width=width,
            height=height,
        )
        fig.update_xaxes(tickmode="linear")
        fig.update_layout(
            barmode="stack",
            xaxis={"categoryorder": "total descending"},
            hovermode="closest",
        )
        return fig


def combine_projects(
    reference_vrd: VRDProject, predict_vrd: VRDProject, name: str
) -> VRDProject:
    """Combine two projects to compare the videos in one to the calculated fingerprints in another;
    essentially merging them.

    The frame extractors will be merged to contain both, and a distance 
    list calculated. The distance list will be changed to convert the 
    indexes to the new indexes of the predicted frames.

    The combined project which is returned from this function should not be 
    worked upon, i.e. no rebuilding of databases and so on.

    Args:
        reference_vrd (VRDProject): The source VRD Project, which contains the fingerprints
        predict_vrd (VRDProject): The prediction set, which need only contain videos
        name (str): The name of the new project, which determines the output directory 

    Returns:
        VRDProject: The combined project
    """
    combined = VRDProject(
        name=name,
        project_base_path=predict_vrd.project_base_path,
        video_path=predict_vrd.video_path,
        network=predict_vrd.network,
        additional_video_extensions=predict_vrd.additional_video_extensions,
    )
    new_frames = predict_vrd.frame_extractor.copy()
    new_frames.all_images = (
        reference_vrd.frame_extractor.all_images
        + predict_vrd.frame_extractor.all_images
    )

    new_frames.initialize_indexes()
    combined.frame_extractor = new_frames

    distance_list = faiss_helper.calculate_distance_list_external(
        external_database_file=predict_vrd.database_file,
        reference_faiss_index=reference_vrd.faiss_index,
        external_frames=predict_vrd.frame_extractor,
        neighbour_num=predict_vrd.neighbours_considered,
        batch_directory=combined.neighbour_batch_path,
    )

    idx_diff = len(reference_vrd.frame_extractor.all_images)
    for expected_idx in range(0, len(distance_list)):

        d, i = distance_list[expected_idx]
        if i[0] != expected_idx + idx_diff:
            d = np.insert(d, 0, 0)
            i = np.insert(i, 0, expected_idx + idx_diff)
            distance_list[expected_idx] = (d, i)

    new_neighbours = Neighbours(new_frames, distance_list)
    combined_neighbours = new_neighbours

    combined.neighbours = combined_neighbours
    return combined
