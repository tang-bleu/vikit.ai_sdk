import asyncio
from loguru import logger
import os
from vikit.common.file_tools import download_or_copy_file, is_valid_path
from vikit.video.video import Video
from vikit.video.video_build_settings import VideoBuildSettings


class LocalEngine:

    def __init__(self, build_settings: VideoBuildSettings = VideoBuildSettings()):
        self.build_settings = build_settings

    def generate(self, video: Video) -> Video:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # If there's already a running loop, create a new task and wait for it
            return loop.create_task(self.generate_async(video))
        else:
            # If no loop is running, use asyncio.run
            return asyncio.run(self.generate_async(video))

    async def generate_async(self, video: Video):
        """
        Build the video in the child classes, unless the video is already built, in  which case
        we just return ourselves (Video gets immutable once generated)

        This is a template method, the child classes should implement the get_handler_chain method

        Args:
            build_settings (VideoBuildSettings): The settings to use for building the video

        Returns:
            Video: The built video

        """
        if video._is_video_built:
            logger.info(f"Video {video.id} is already built, returning it")
            return video

        current_dir = os.getcwd()
        logger.trace(
            f"Starting the pre build hook for Video {video.id}, current_dir {current_dir} "
        )

        wfolder_changed = _set_working_folder_dir(self.build_settings.output_path)
        logger.trace(f"Working folder has changed? : {wfolder_changed}")

        logger.trace(
            f"Starting the pre build hook for Video {video.id} of type {video.short_type_name} / {type(video)}"
        )
        await video.run_pre_build_actions_hook(build_settings=self.build_settings)

        built_video = None
        if not video.are_build_settings_prepared:
            video.build_settings = self.build_settings
            video._source = type(
                self.build_settings.get_ml_models_gateway()  # TODO: this is hacky and should be refactored
                # so that we infer source from the different handlers (initial video generator, interpolation, etc)
            ).__name__  # as the source(s) of the video is used later to decide if we need to reencode the video

            await video.prepare_build(build_settings=self.build_settings)
            video.are_build_settings_prepared = True

        logger.info(f"Starting the building of Video {video.id} ")

        built_video = await video.run_build_core_logic_hook(
            build_settings=self.build_settings
        )  # logic from the child classes if any
        built_video = await self._gather_and_run_handlers(video)

        logger.debug(f"Starting the post build hook for Video {video.id} ")
        await video.run_post_build_actions_hook(build_settings=self.build_settings)

        if self.build_settings.target_file_name:
            video.set_final_video_name(
                output_file_name=self.build_settings.target_file_name,
            )

        if wfolder_changed:
            # go back to the original working folder
            _set_working_folder_dir(current_dir)

        video.is_video_built = True

        return built_video

    async def _gather_and_run_handlers(self, video: Video) -> Video:
        """
        Gather the handler chain and run it
        """
        logger.trace("Gathering the handler chain")
        built_video = None

        handler_chain = video._get_and_initialize_video_handler_chain(
            build_settings=self.build_settings
        )
        if not handler_chain:
            logger.warning(
                f"No handler chain defined for the video of type {video.short_type_name}"
            )
        else:
            logger.debug(
                f"about to run {len(handler_chain)} handlers for video {video.id} of type {video.short_type_name} / {type(video)}"
            )
            for handler in handler_chain:
                built_video = await handler.execute_async(video)
                built_video.is_video_built = True

                assert built_video.media_url, "The video media URL is not set"

        video.metadata.title = video.get_title()
        video.media_url = await download_or_copy_file(
            url=video.media_url,
            local_path=video.get_file_name_by_state(video.build_settings),
        )
        video.metadata.duration = (
            video.get_duration()
        )  # This needs to happen once the video has been downloaded

        return built_video


def _set_working_folder_dir(working_folder_path: str):
    if working_folder_path:
        if is_valid_path(working_folder_path):
            os.chdir(working_folder_path)
            return True
        else:
            logger.warning(
                f"Video target dir path name is None or invalid, using the current video generation path: {os.getcwd()}"
            )
            return False
