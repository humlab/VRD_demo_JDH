"""A module for low-memory sequence finding based on a neighbour distance list


"""
import sys
import time

import numpy as np
import pandas as pd
import PIL
from IPython.core.display import display
from scipy.sparse import lil_matrix, triu
from tqdm.auto import tqdm

from vrd.vrd_project import VRDProject

from . import frame_extractor, neighbours

sys.path.append("..")


class SequenceFinder:
    """Finds sequences (frames matching second-per-second) from a given neighbours class."""

    sequence_lil_matrix: lil_matrix
    shortest_allowed_sequence: int
    max_distance: int
    neigh: neighbours.Neighbours

    def __init__(
        self,
        neigh: neighbours.Neighbours,
        max_distance=30000,
        shortest_allowed_sequence=3,
    ) -> None:
        self.neigh = neigh
        self.max_distance = max_distance
        self.shortest_allowed_sequence = shortest_allowed_sequence

        matrix = self._calculate_sequence_matrix(max_distance=max_distance)
        # Make upper triangular
        matrix = triu(matrix + matrix.T)
        self.sequence_lil_matrix_distance = matrix.tolil()
        self.sequence_lil_matrix = self.sequence_lil_matrix_distance.astype(np.bool8)

        # self._remove_from_same_video()

        self.found_sequences = None

    def filter_matches_from_same_video(self):
        """Filters all frame matches for the same video against itself,
        as these matches are common (scenes largely unchanged from the last second and so on).
        """
        print("Removing neighbours from same video...")
        before = self.sequence_lil_matrix.nnz
        for idxes in tqdm(self.neigh.frames.cached_video_index.values()):
            arr = np.array(list(idxes))
            amin = np.min(arr)
            amax = np.max(arr) + 1  # Include last element
            self.sequence_lil_matrix[amin:amax, amin:amax] = False
        after = self.sequence_lil_matrix.nnz
        print(
            f"Removed {before-after} entries ({(((before-after)*100) / before):.2f}%)!"
        )

    def _calculate_sequence_matrix(self, max_distance=30000):
        """Create a sparse matrix from the neighbour distance matrix.
        This sparse matrix consists of bool values, and the actual distance is currently lost.
        Only distances belov the specified value will be considered.

        The matrix is of the size (len(frames.all_frames), len(frames.all_frames))

        Args:
            max_distance (int, optional): The maximum distance to keep. Defaults to 30000.

        Returns:
            A scipy.lil_matrix containing all neighbours that were within the specified distance.
        """
        dlist = self.neigh.distance_list
        max_index = np.max(np.array([np.max(i) for d, i in dlist]))
        print(len(self.neigh.frames.all_images))
        print(f"Maximum index found: {max_index}")

        neighbour_matrix = lil_matrix((max_index + 1, max_index + 1), dtype=np.float32)

        for d, i in tqdm(dlist):
            self_index = i[0]
            for idx_below_max_distance in (
                np.where(d[1:] < max_distance)[0] + 1
            ):  # Index (excluding the first) where value is below max_distance
                neighbour_matrix[self_index, i[idx_below_max_distance]] = d[
                    idx_below_max_distance
                ]

        print(f"Number of bytes required: {neighbour_matrix.data.nbytes}")
        return neighbour_matrix

    @staticmethod
    def _find_sequence(lst: list, shortest_sequence=3, allow_skip=0):
        """Helper method to find sequences in a list of integers.

        Args:
            lst (list): A list of integers
            shortest_sequence (int, optional): Shortest sequence to save. Defaults to 3.
            allow_skip (int, optional): The number of skips in a sequence to allow. Defaults to 0.

        Returns:
            list: A list of lists, where each list contains a found sequence
        """
        if len(lst) < shortest_sequence:
            return None
        arr = np.array(lst)
        diff = arr - np.roll(arr, 1)

        found = []
        all_found = []
        for i, in_sequence in enumerate(diff < (2 + allow_skip)):
            if in_sequence:
                found.append(i)
            else:
                if len(found) >= shortest_sequence:
                    all_found.append(arr[found])
                    found = []
                found.clear()
        if len(found) >= shortest_sequence:
            all_found.append(arr[found])
        # print(all_found)
        if len(all_found) == 0:
            return None
        return all_found

    def find_sequences(self, shortest_sequence=3, allow_skip=0, combine_overlap=False):
        """Look for sequences in the data

        Args:
            shortest_sequence (int, optional): Shortest sequence to consider. Defaults to 3.
            allow_skip (int, optional): Allows frames to be 'skipped'.
                Example, with a skip of 0 only sequential matches are allowed (1,2,3,4,5).
                With a skip of 1, the sequence (1,2,4,5) is a sequence of length 4.
                Defaults to 0.

        Returns:
            list: A list of sequences, in tuples with the contents (start1, start2, duration).

        """
        sparse_matrix = self.sequence_lil_matrix.copy()

        # Shift all rows left depending on row number
        for i in range(0, len(sparse_matrix.rows)):
            sparse_matrix.rows[i] = [x - i for x in sparse_matrix.rows[i]]

        col = sparse_matrix.T  # work on columns instead!

        found_sequences = []
        for i in range(0, len(col.rows)):
            if len(col.rows[i]) > 0:
                found = self._find_sequence(
                    col.rows[i],
                    shortest_sequence=shortest_sequence,
                    allow_skip=allow_skip,
                )
                if found is not None:
                    found_sequences.append((i, found))

        # Convert to start-stop-duration tuple instead of list
        converted_sequences = []
        for start_time, sequences in found_sequences:
            if sequences is None:
                continue
            for seq in sequences:
                start, stop = seq[0], seq[-1]

                converted_sequences.append(
                    (start_time + start, start, stop - start + 1)
                )

        converted_sequences = sorted(
            converted_sequences, key=lambda x: x[2], reverse=True
        )
        if combine_overlap:
            return sorted(
                SequenceFinder.combine_overlap(converted_sequences),
                key=lambda x: x[2],
                reverse=True,
            )
        return converted_sequences

    @staticmethod
    def _combine_overlapping_intervals(intervals):
        """Combines intervals by looking for start/end.

        Can be further improved by using 1,-1 instead of true/false for is_start
        and using np.cumsum and np.where, but it's hardly necessary.

        Args:
            intervals (_type_): Iterable containing tuples of indexes in the form of (start, end)

        Returns:
            list: A list where the overlaps in the input has been combined
        """

        indexes = []
        is_start = []
        for start, end in intervals:
            indexes.extend([start, end])
            is_start.extend([True, False])

        indexes = np.array(indexes, dtype=np.int32)
        is_start = np.array(is_start, dtype=np.bool8)

        sort_order = np.argsort(indexes)
        indexes = indexes[sort_order]
        is_start = is_start[sort_order]

        start_count = 0
        start_index = -1
        merged_ranges = []
        for idx in range(len(indexes)):
            if is_start[idx]:
                if start_count == 0:
                    start_index = indexes[idx]
                start_count += 1
            else:
                start_count -= 1
                if start_count == 0:
                    # We found the end of a range
                    merged_ranges.append((start_index, indexes[idx]))
        return merged_ranges

    @staticmethod
    def combine_overlap(sequences):
        """Convert sequences to sequences without overlap.

        This is done by merging the sequences that overlap, per video.

        Due to the nature of overlap, this can cause a small shift in overlap

        Args:
            sequences (_type_): _description_

        Returns:
            _type_: _description_
        """
        s1_orig_intervals = [[s, s + dur - 1] for s, e, dur in sequences]
        s1_orig_intervals_arr = np.array(s1_orig_intervals)

        s1_intervals = SequenceFinder._combine_overlapping_intervals(s1_orig_intervals)
        # print(s1_intervals)

        all_sequnces_no_overlap = {}
        for start, end in s1_intervals:
            relevant_sequences = (s1_orig_intervals_arr[:, 0] >= start) & (
                s1_orig_intervals_arr[:, 1] <= end
            )

            s2_orig_intervals = [
                [e, e + dur - 1]
                for s, e, dur in [
                    sequences[x] for x in list(np.where(relevant_sequences)[0])
                ]
            ]
            all_sequnces_no_overlap[
                (start, end)
            ] = SequenceFinder._combine_overlapping_intervals(s2_orig_intervals)

        # print(all_sequnces_no_overlap)
        new_sequence = []
        for seq1, seq2_list in all_sequnces_no_overlap.items():
            s1_dur = seq1[1] - seq1[0]
            for seq2 in seq2_list:
                s2_dur = seq2[1] - seq2[0]
                new_sequence.append((seq1[0], seq2[0], np.min([s1_dur, s2_dur])))
        print(f"Merged from {len(sequences)} to {len(new_sequence)} sequences")
        return new_sequence



    def get_sequence_dataframe(self, sequences: list) -> pd.DataFrame:
        """Creates a Pandas dataframe with all sequences.
        This includes:
        * Video names
        * Video start times
        * Duration
        * Original indexes
        * Mean distance
        * Number of non-matching frames

        This DataFrame can be shown in a notebook, and possibly filtered, to be recreated into a seqency by the
        dataframe_to_sequence method.

        Args:
            sequences (list): _description_

        Returns:
            pd.DataFrame: _description_
        """
        df_list = []
        for start1, start2, duration in sequences:
            v1, v1_start = self.neigh.frames.get_video_and_start_time_from_index(start1)
            v2, v2_start = self.neigh.frames.get_video_and_start_time_from_index(start2)

            time_str = lambda x: time.strftime("%H:%M:%S", time.gmtime(x))
            
            dist_mean, dist_zeros = self.get_sequence_mean_distance(start1, start2, duration)

            df_list.append(
                {
                    "Video 1": v1,
                    "Video 2": v2,
                    "Video 1 Start time": time_str(v1_start),
                    "Video 1 End time": time_str(v1_start + duration),
                    "Video 2 Start time": time_str(v2_start),
                    "Video 2 End time": time_str(v2_start + duration),
                    "Duration": duration,
                    "Index Video 1": start1,
                    "Index Video 2": start2,
                    "Match mean distance": dist_mean,
                    "# not matching": dist_zeros,
                }
            )
        return pd.DataFrame(df_list)
    
    @staticmethod
    def dataframe_to_sequence(df: pd.DataFrame) -> list:
        new_seq = []
        for id, row in df.to_dict('index').items():
            new_seq.append((row['Index Video 1'], row['Index Video 2'], row['Duration']))
        return new_seq
    

    def show_notebook_sequence(
        self,
        sequences,
        show_limit: int = None,
        show_shift=False,
        frame_resize=(30, 30),
        sort_order: callable = None,
    ):
        """Displays longest-to-shortest sequences in a notebook

        Args:
            sequences (list): The sequence list of tuples (start1, start2, duration)
            show_limit (int, optional): The maximum number of sequences to show (Default: all)
            show_shift (bool, optional): Show statistics for "shifting", i.e. moving the matches one or 
                several seconds forward or backwards.
            frame_resize (tuple, optional): Size of the thumbnails of each frame
        """
        frames = self.neigh.frames

        if sort_order is None:
            # Default: Sort by duration
            sequences = sorted(sequences, key=lambda x: x[2], reverse=True)
        else:
            sequences = sort_order(sequences)

        time_str = lambda x: time.strftime("%H:%M:%S", time.gmtime(x))
        for start1, start2, duration in sequences[:show_limit]:
            v1, v1_start = frames.get_video_and_start_time_from_index(start1)
            v2, v2_start = frames.get_video_and_start_time_from_index(start2)

            # duration = duration + 1

            if show_shift:
                self.show_shift_df(start1, start2, duration)

            dist_mean, dist_zeros = self.get_sequence_mean_distance(
                start1, start2, duration
            )

            pd_data = [
                {
                    "Video": v1,
                    "Start time": time_str(v1_start),
                    "End time": time_str(v1_start + duration),
                    "Duration": duration,
                    "Index": start1,
                    "Match mean distance": dist_mean,
                    "# not matching": dist_zeros,
                },
                {
                    "Video": v2,
                    "Start time": time_str(v2_start),
                    "End time": time_str(v2_start + duration),
                    "Duration": duration,
                    "Index": start2,
                    "Match mean distance": dist_mean,
                    "# not matching": dist_zeros,
                },
            ]
            display(pd.DataFrame(pd_data).set_index("Video"))

            display(
                self.show_sequence(
                    start1, start2, duration, frames, frame_resize=frame_resize
                )
            )

    def get_sequence_mean_distance(self, start1, start2, duration):
        res_arr = np.zeros(duration, dtype=np.float32)
        for i in range(duration):
            res_arr[i] = self.sequence_lil_matrix_distance[start2 + i, start1 + i]
        nonzeros = res_arr[np.where(res_arr > 0)[0]]
        dist_mean = np.mean(nonzeros) if len(nonzeros) > 0 else np.nan
        dist_zeros = np.sum(res_arr == 0)
        return dist_mean, dist_zeros

    def show_shift_df(self, start1, start2, duration):
        shift_results = []
        for shift in range(-3, 4):
            res_arr = np.zeros(duration, dtype=np.float32)
            for i in range(duration):
                res_arr[i] = self.sequence_lil_matrix_distance[
                    start2 + i + shift, start1 + i
                ]
            nonzeros = res_arr[np.where(res_arr > 0)[0]]

            dist_mean = np.mean(nonzeros) if len(nonzeros) > 0 else np.nan
            dist_zeros = np.sum(res_arr == 0)
            shift_results.append(
                {"Shift": shift, "Mean": dist_mean, "Zeros": dist_zeros}
            )
        df = pd.DataFrame(shift_results).set_index("Shift")
        display(df)
        # if minimize_nonmatch:
        #     df = df[df.Zeros == df.Zeros.min()]
        #     if len(df > 1):
        #         df = df[df.Mean == df.Mean.min()]
        #     start1 -= df.reset_index().Shift.values[0]

    @staticmethod
    def remove_unwanted_sequences(sequences, project: VRDProject, excel_file: str):
        """ Removes unwanted sequences. The unwanted sequences are defined in an excel file,
        containing the columns:
                'Video': The video name
                'Start time': The start time in a string, formatted as HH:MM:SS 
                'Duration': An integer describing the duration of the unwanted sequence, in seconds
        
        
        Note: Only the time stamps in the file is used, as the index itself can change """
        frames = project.frame_extractor
        df = pd.read_excel(excel_file)
        df["start_time_seconds"] = (
            pd.to_timedelta(df["Start time"].astype(str)).dt.total_seconds().apply(int)
        )

        highest_dur = df.sort_values("Duration", ascending=False).drop_duplicates(
            ["Video", "start_time_seconds"]
        )
        highest_dur = highest_dur.sort_values(by=["Video", "start_time_seconds"])

        remove_set = set()
        for vid in df.Video.unique():
            vid_df = df[df.Video == vid]
            video_frames = sorted(list(frames.get_index_from_video_name(vid)))
            for time, duration in vid_df[["start_time_seconds", "Duration"]].values:
                remove_set.update(
                    list(range(video_frames[time], video_frames[time + duration]))
                )
        sequence_remove_indexes = []
        for seq_index, (start1, start2, dur) in enumerate(sequences):
            seq_to_find = list(range(start1, start1 + dur)) + list(
                range(start2, start2 + dur)
            )
            for idx in seq_to_find:
                if idx in remove_set:
                    sequence_remove_indexes.append(seq_index)
                    break
        for seq_index in sorted(sequence_remove_indexes, reverse=True):
            del sequences[seq_index]

    @staticmethod
    def show_sequence(
        start1: int,
        start2: int,
        duration: int,
        frames: frame_extractor.FrameExtractor,
        frame_resize=(200, 200),
    ):
        """Generates a comparison image for a given sequence

        Args:
            start1 (int): Start time 1 (as related to the Frame)
            start2 (int): _description_
            duration (int): _description_
            frames (FrameExtractor): _description_
            frame_resize (tuple, optional): Size of each resulting image in the sequence. Defaults to (200, 200).

        Returns:
            A PIL image with the whole sequence
        """

        images = frames.all_images

        merged = PIL.Image.new(
            "RGB", (frame_resize[0] * duration, frame_resize[1] * 2), (250, 250, 250)
        )

        for i in range(0, duration):
            img1 = PIL.Image.open(images[start1 + i]).resize(frame_resize)
            img2 = PIL.Image.open(images[start2 + i]).resize(frame_resize)
            merged.paste(img1, (frame_resize[0] * i, 0))
            merged.paste(img2, (frame_resize[0] * i, frame_resize[1]))
        return merged
