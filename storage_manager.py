import os
import config
import logging
from time import time


class StorageManager:
    def __init__(self) -> None:
        if not os.path.exists(config.PROCESSED_VIDEOS_DIR):
            os.makedirs(config.PROCESSED_VIDEOS_DIR)
        else:
            self.clear_processed_videos()

        if not os.path.exists(config.PROCESSED_IMAGES_DIR):
            os.makedirs(config.PROCESSED_IMAGES_DIR)
        else:
            self.clear_processed_images()

        if not os.path.exists(config.VIDEO_THUMBNAILS_DIR):
            os.makedirs(config.VIDEO_THUMBNAILS_DIR)
        else:
            self.clear_processed_thumbnails()

    def get_image_filepath(self) -> str:
        image_filename = config.IMAGE_FILENAME.format(processed_time=int(time()))
        image_filepath = os.path.join(config.PROCESSED_IMAGES_DIR, image_filename)
        return image_filepath, image_filename

    def get_video_filepath(self) -> str:
        video_filename = config.VIDEO_FILENAME.format(
            processed_time=int(time()),
        )
        video_filepath = os.path.join(
            config.PROCESSED_VIDEOS_DIR,
            video_filename,
        )
        return video_filepath, video_filename

    def get_thumbnail_filepath(self) -> str:
        thumbnail_filename = config.THUMBNAIL_FILENAME.format(
            processed_time=int(time()),
        )
        thumbnail_filepath = os.path.join(
            config.VIDEO_THUMBNAILS_DIR,
            thumbnail_filename,
        )
        return thumbnail_filepath, thumbnail_filename

    def clear_processed_videos(self) -> None:
        for filename in os.listdir(config.PROCESSED_VIDEOS_DIR):
            file_path = os.path.join(config.PROCESSED_VIDEOS_DIR, filename)
            try:
                if os.path.isfile(file_path):
                    creation_time = os.path.getctime(file_path)
                    current_time = time()
                    file_age = current_time - creation_time
                    if file_age > 1000:
                        os.remove(file_path)
                        logging.info(f"Deleted file: {file_path}")
                    else:
                        logging.info(f"File is kept: {file_path}")
            except Exception as e:
                logging.error(f"Error deleting file {file_path}: {e}")

    def clear_processed_images(self) -> None:
        for filename in os.listdir(config.PROCESSED_IMAGES_DIR):
            file_path = os.path.join(config.PROCESSED_IMAGES_DIR, filename)
            try:
                if os.path.isfile(file_path):
                    creation_time = os.path.getctime(file_path)
                    current_time = time()
                    file_age = current_time - creation_time
                    if file_age > 1000:
                        os.remove(file_path)
                        logging.info(f"Deleted file: {file_path}")
                    else:
                        logging.info(f"File is kept: {file_path}")
            except Exception as e:
                logging.error(f"Error deleting file {file_path}: {e}")

    def clear_processed_thumbnails(self) -> None:
        for filename in os.listdir(config.VIDEO_THUMBNAILS_DIR):
            file_path = os.path.join(config.VIDEO_THUMBNAILS_DIR, filename)
            try:
                if os.path.isfile(file_path):
                    creation_time = os.path.getctime(file_path)
                    current_time = time()
                    file_age = current_time - creation_time
                    if file_age > 1000:
                        os.remove(file_path)
                        logging.info(f"Deleted file: {file_path}")
                    else:
                        logging.info(f"File is kept: {file_path}")
            except Exception as e:
                logging.error(f"Error deleting file {file_path}: {e}")
