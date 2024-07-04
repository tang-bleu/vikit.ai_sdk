import pytest
import unittest
import warnings
import datetime
import uuid

from vikit.common.context_managers import WorkingFolderContext

from vikit.video.raw_text_based_video import RawTextBasedVideo
from vikit.video.video_types import VideoType
from vikit.video.video_build_settings import VideoBuildSettings
from vikit.video.video_file_name import VideoFileName
from vikit.video.prompt_based_video import PromptBasedVideo
from vikit.video.composite_video import CompositeVideo
from vikit.prompt.prompt_factory import PromptFactory
from vikit.video.transition import Transition
from vikit.video.video_metadata import VideoMetadata


class TestVideoFileNames(unittest.TestCase):
    def setUp(self) -> None:
        warnings.simplefilter("ignore", category=ResourceWarning)
        warnings.simplefilter("ignore", category=UserWarning)

    @pytest.mark.unit
    def test_nominal_file_name(self):
        # Create a VideoFileName instance with sample values

        bld_set = VideoBuildSettings()
        bld_set._id = "1234567890"
        bld_set._build_date = datetime.date(2024, 7, 1)
        bld_set._build_time = datetime.time(23, 4, 42)

        video_file_name = VideoFileName(
            build_settings=bld_set,
            video_metadata=VideoMetadata(
                title="Sample Video",
            ),
            video_type="comproot",
            video_features="dorio",
        )

        # Check if the file name is generated correctly
        expected_file_name = (
            "Sample Video_comproot_dorio_1234567890_2024-07-01_23:04:42_UID_"
            + str(video_file_name.unique_id)
            + ".mp4"
        )
        self.assertEqual(video_file_name.file_name, expected_file_name)

    @pytest.mark.unit
    def test_none_input(self):
        with self.assertRaises(ValueError):
            _ = VideoFileName(build_settings=None, video_metadata=None)

    @pytest.mark.unit
    def test_raw_text_video_file_name(self):
        """
        Test if the file name of a RawTextBasedVideo instance is generated correctly

        Here we expect the name to conform to the general video file name format.
        """
        raw_text_video = RawTextBasedVideo(raw_text_prompt="test prompt")

        fname = str(
            raw_text_video.get_file_name_by_state(build_settings=VideoBuildSettings()),
        )
        assert VideoFileName.is_video_file_name(
            fname
        ), "The file name of the RawTextBasedVideo instance is not valid. Generated file name: {}".format(
            fname
        )

    @pytest.mark.unit
    def test_from_file_name(self):
        id = str(uuid.uuid4())
        file_name = (
            f"exampletitle_comproot_ooooo_1234567890_2022-01-01_12:00_UID_{id}.mp4"
        )
        video_file_name = VideoFileName.from_file_name(file_name=file_name)
        assert video_file_name.title == "exampletitle"
        assert video_file_name.video_type == "comproot"
        assert video_file_name.video_features == "ooooo"
        assert video_file_name.build_id == "1234567890"
        assert video_file_name._build_date == datetime.date(2022, 1, 1)
        assert video_file_name._build_time == datetime.time(12, 0)
        assert video_file_name.unique_id == uuid.UUID(id)

    @pytest.mark.unit
    def test_extract_features(self):

        bld_set = VideoBuildSettings()
        bld_set._id = "1234567890"
        bld_set._build_date = datetime.date(2024, 7, 1)
        bld_set._build_time = datetime.time(23, 4, 42)

        video_file_name = VideoFileName(
            video_metadata=VideoMetadata(title="Sample Video"),
            build_settings=bld_set,
            video_type="comproot",
            video_features="dogrs",
            file_extension="mp4",
        )
        # Extract the features
        extracted_features = video_file_name.extract_features_as_string()
        # Check if the extracted features are correct
        self.assertEqual(extracted_features, "dogrs")

    @pytest.mark.unit
    def test_extract_features_no_features(self):

        bld_set = VideoBuildSettings()
        bld_set._id = "1234567890"
        bld_set._build_date = datetime.date(2024, 7, 1)
        bld_set._build_time = datetime.time(23, 4, 42)
        bld_set.music_building_context.apply_background_music = False
        bld_set.interpolate = False

        video_file_name = VideoFileName(
            video_metadata=VideoMetadata(title="Sample Video"),
            build_settings=bld_set,
            video_type="comproot",
            file_extension="mp4",
        )
        # Extract the features
        extracted_features = video_file_name.extract_features_as_string()
        # Check if the extracted features are correct
        expected_features = "ooooo"
        self.assertEqual(extracted_features, expected_features)

    def get_test_build_settings(self):
        bld_set = VideoBuildSettings()
        bld_set.test_mode = True
        bld_set._id = "1234567890"
        bld_set._build_date = datetime.date(2024, 7, 1)
        bld_set._build_time = datetime.time(23, 4, 42)
        bld_set.output_path = "output_path"
        bld_set.include_audio_subtitles = False
        bld_set.interpolate = False
        bld_set.music_building_context.apply_background_music = True
        bld_set.music_building_context.generate_background_music = True

        return bld_set

    @pytest.mark.unit
    def test_video_file_names_across_building_steps(self):

        with WorkingFolderContext():
            root_composite_video = CompositeVideo()
            bld_set = self.get_test_build_settings()

            fname = str(
                root_composite_video.get_file_name_by_state(build_settings=bld_set)
            )
            vid_fname = VideoFileName.from_file_name(fname)

            assert VideoFileName.is_video_file_name(
                fname
            ), "The file name of the video instance is not valid. Generated file name: {}".format(
                fname
            )
            assert (
                vid_fname.video_features == "ooooo"
            ), "The video features are not correct: features returned: {}".format(
                vid_fname.video_features
            )
            assert (
                vid_fname.build_id == bld_set.id
            ), "The build ID is not correct, {}".format(bld_set.id)

            assert vid_fname._build_date == bld_set.build_date
            assert vid_fname._build_time == bld_set.build_time
            assert vid_fname.unique_id is not None

            # now we stop to the next video building step (build the video, here we decide to
            # interpolate and reencode), we expect the video features to be updated accordingly
            root_composite_video.metadata.is_video_generated = True
            root_composite_video.metadata.is_interpolated = True
            root_composite_video.metadata.is_reencoded = True

            fname_built = str(
                root_composite_video.get_file_name_by_state(build_settings=bld_set)
            )
            vid_fname = VideoFileName.from_file_name(fname_built)
            assert VideoFileName.is_video_file_name(
                fname
            ), "The file name of the video instance is not valid. Generated file name: {}".format(
                fname
            )
            assert (
                vid_fname.video_features == "oorio"
            ), "The video features are not correct: features returned: {}".format(
                vid_fname.video_features
            )
            assert (  # Check the build id has not changed
                vid_fname.build_id == bld_set.id
            ), "The build ID is not correct, {}".format(bld_set.id)

            # now we stop to the next video building step, here we add background music, and it needs
            # to be generated. We expect the video features to be updated accordingly
            root_composite_video.metadata.is_bg_music_applied = True
            root_composite_video.metadata.is_bg_music_generated = True
            fname_built = str(
                root_composite_video.get_file_name_by_state(build_settings=bld_set)
            )
            vid_fname = VideoFileName.from_file_name(fname_built)
            assert VideoFileName.is_video_file_name(
                fname
            ), "The file name of the video instance is not valid. Generated file name: {}".format(
                fname
            )
            assert (
                vid_fname.video_features == "gorio"
            ), "The video features are not correct: features returned: {}".format(
                vid_fname.video_features
            )
            assert (  # Check the build id has not changed
                vid_fname.build_id == bld_set.id
            ), "The build ID is not correct, {}".format(bld_set.id)

            # now we stop to the next video building step, here we add read aloud subtitles.
            #  We expect the video features to be updated accordingly
            root_composite_video.metadata.is_prompt_read_aloud = True
            fname_built = str(
                root_composite_video.get_file_name_by_state(build_settings=bld_set)
            )
            vid_fname = VideoFileName.from_file_name(fname_built)
            assert VideoFileName.is_video_file_name(
                fname
            ), "The file name of the video instance is not valid. Generated file name: {}".format(
                fname
            )
            assert (
                vid_fname.video_features == "gvrio"
            ), "The video features are not correct: features returned: {}".format(
                vid_fname.video_features
            )
            assert (  # Check the build id has not changed
                vid_fname.build_id == bld_set.id
            ), "The build ID is not correct, {}".format(bld_set.id)