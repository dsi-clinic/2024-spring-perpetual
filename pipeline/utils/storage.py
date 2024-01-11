"""Utilities for reading and writing files across data stores.
"""

import io
import os
import tempfile

from abc import ABC, abstractmethod
from contextlib import contextmanager
from constants import DATA_DIR
from typing import Iterator, Optional


class IDataReader(ABC):
    """Abstract class for reading files from a data store.
    """

    @abstractmethod
    @contextmanager
    def read_file(self, name: str) -> io.IOBase:
        """Reads a file from a data store. Acts as a context manager using
        the [contextlib](https://docs.python.org/3/library/contextlib.html)
        package from the Python Standard Library.

        Args:
            name (`str`): The full name of the file (i.e., the
                absolute path within the data store).

        Yields:
            (`io.IOBase`): The file object.
        """
        raise NotImplementedError


class LocalDataReader(IDataReader):
    """Class for reading files from a local directory.
    """

    @contextmanager
    def read_file(self, name: str) -> Iterator[io.IOBase]:
        """Reads a file from a local data store. Acts as a context manager using
        the [contextlib](https://docs.python.org/3/library/contextlib.html)
        package from the Python Standard Library.

        Args:
            name (`str`): The full name of the file (i.e., the
                absolute path within the data store).

        Yields:
            (`io.IOBase`): The file object.
        """
        # Define file path
        fpath = DATA_DIR / name

        # Read first few bytes
        with open(fpath, "rb") as f:
            first_bytes = f.read(3)
        
        # Detect UTF-8 BOM encoding from bytes
        encoding = "utf-8-sig" if first_bytes == b'\xef\xbb\xbf' else None

        # Yield file object
        f = open(fpath, encoding=encoding)
        try:
            yield f
        finally:
            f.close()


class GoogleCloudStorageReader(IDataReader):
    """Concrete class for accessing Google Cloud Storage."""

    def __init__(self) -> None:
        """Initializes a new instance of a `GoogleCloudStorageHelper`.

        Raises:
            `RuntimeError` if an environment variable, 
                `GOOGLE_CLOUD_STORAGE_BUCKET`, is not found.

        Args:
            `None`

        Returns:
            `None`
        """
        # Parse environment variables
        try:
            bucket_name = os.environ["GOOGLE_CLOUD_STORAGE_BUCKET"]
        except KeyError as e:
            raise RuntimeError(
                "Failed to initialize GoogleCloudStorageHelper."
                f"Missing expected environment variable \"{e}\"."
            ) from None
        
        # Set up reference to Cloud Storage bucket
        from google.cloud import storage
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(bucket_name)


    @contextmanager
    def open_file(self, filename: str) -> Iterator[io.IOBase]:
        """Opens a file with the given name and mode.

        Args:
            filename (`str`): The file name (i.e., key), representing
                the path to the file within the data bucket
                (e.g., "states.geoparquet").

            mode (`str`): The file opening method. Defaults to
                reading text (i.e., "r").

        Yields:
            (`io.IOBase`): A file object.
        """
        # Fetch reference to Cloud Storage blob in bucket
        blob = self.bucket.blob(filename)

        # Stream blob to temporary file on disk
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            blob.download_to_filename(tf.name)
            temp_filename = tf.name

        # Read first few bytes
        with open(temp_filename, "rb") as f:
            first_bytes = f.read(3)

        # Detect UTF-8 BOM encoding from bytes
        encoding = "utf-8-sig" if first_bytes == b'\xef\xbb\xbf' else None
        
        # Yield file object
        f = open(temp_filename, encoding=encoding)
        try:
            yield f
        finally:
            f.close()
            os.remove(temp_filename)


class IDataReaderFactory:
    """Factory for fetching a Singleton instance
    of an `IDataReader` based on environment.
    """

    _helper: Optional[IDataReader] = None

    @staticmethod
    def get() -> IDataReader:
        """Fetches a local or cloud-based file system
        helper based on the current name of the
        development environment (e.g., "DEV" or "PROD").

        Args:
            `None`

        Returns:
            (`IDataReader`)
        """
        if not IDataReaderFactory._helper:
            env = os.environ.get("ENV", "DEV")

            if env == "DEV":
                IDataReaderFactory._helper = LocalDataReader()
            elif env == "PROD":
                IDataReaderFactory._helper = GoogleCloudStorageReader()
            else:
                raise RuntimeError(
                    "Unable to instantiate data reader. Invalid "
                    f"environment variable passed for 'ENV': {env}."
                )
        return IDataReaderFactory._helper
