"""DataCore class."""

import ast
from pathlib import Path
from typing import Any, Dict, Optional, TypeVar, Union, cast

import numpy as np
import pandas as pd
import torch
from pyarrow import timestamp
from safetensors.torch import load_file, save_file
from torchgen.utils import FileManager

# Type aliases for clarity
DataFrame = TypeVar("DataFrame", bound="pd.DataFrame")
NDArray = TypeVar("NDArray", bound="np.ndarray")


def save_tensor_dict(
    tensor_dict: Dict[str, torch.Tensor], file_path: Union[str, Path]
) -> Dict[str, Any]:
    """
    Save a dictionary of tensors to a file using safetensors format.

    This function provides a safer and more efficient alternative to PyTorch's
    native save function, ensuring tensors are stored in a secure format.

    Args:
        tensor_dict: Dictionary mapping names to torch tensors
        file_path: Path where the tensors will be saved

    Returns:
        Dictionary containing metadata about the saved tensors
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)

    # Ensure the parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Save tensors using safetensors format and return the metadata
    return save_file(tensor_dict, str(file_path))


def load_tensor_dict(file_path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    """
    Load tensors from a safetensors file.

    This function provides a safer and more efficient alternative to PyTorch's
    native load function, ensuring tensors are loaded securely.

    Args:
        file_path: Path to the safetensors file

    Returns:
        Dictionary mapping names to loaded tensors
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)

    # Ensure the file exists
    if not file_path.exists():
        raise FileNotFoundError(f"Tensor file not found at {file_path}")

    # Load tensors from safetensors format and return them
    return load_file(str(file_path))


class DataFrameProcessor:
    """
    The DataFrameProcessor class provides utilities for processing and analyzing
    data stored in pandas DataFrames. It serves as a foundational tool for data
    cleaning, transformation, and computation tasks that are common in data
    analysis workflows.

    This class is designed to encapsulate a series of standardized operations
    to simplify repetitive tasks involving DataFrame manipulation. Users can
    leverage this class to streamline their code and enhance consistency when
    working with tabular data.

    :ivar dataframe: The pandas DataFrame to be processed by the class instance.
    :type dataframe: pandas.DataFrame
    :ivar config: Configuration settings for processing operations, stored as
        a dictionary. It may include parameters for handling missing data,
        computation thresholds, and other processing options.
    :type config: dict
    """

    def __init__(self) -> None:
        self.dataframe: Optional[DataFrame] = None
        self.config: Dict[str, Any] = {
            "missing_value_handling": "drop",
            "thresholds": {
                "numeric_threshold": 0.5,
                "categorical_threshold": 0.5,
                "datetime_threshold": 0.5,
                "text_threshold": 0.5,
            },
        }

    def create_dataframe(self, data: Any) -> DataFrame:
        """
        Creates and returns a Pandas DataFrame from a provided dataset. The method
        is designed to handle structured data input to transform it into tabular format
        for further processing or analysis. The input `data` should typically be a format
        recognizable by Pandas for creating a DataFrame (e.g., dictionary, list of
        dictionaries, or other structured formats).

        :param data: The structured input data to be transformed into a Pandas
            DataFrame.
        :type data: Any
        :return: A Pandas DataFrame constructed from the provided data.
        :rtype: pandas.DataFrame
        """
        self.dataframe = data
        return cast(DataFrame, self.dataframe)


class NumpyProcessor:
    """
    A utility class for processing and performing operations on NumPy arrays.

    This class provides methods to efficiently handle, manipulate, and perform
    computations on NumPy array structures. It is designed for data analysis,
    numerical operations, and other use cases where NumPy arrays are utilized.

    :ivar data: Container for the NumPy array to be processed.
    :type data: numpy.ndarray
    :ivar processed: Boolean flag indicating whether the data has been
        processed.
    :type processed: bool
    """

    data: Optional[NDArray] = None
    processed: bool = False

    def __init__(self) -> None:
        self.data: Optional[NDArray] = None
        self.processed: bool = False

    def generate_matrices(self, df: DataFrame) -> NDArray:
        """Generate matrices from DataFrame.

        :param df: Input DataFrame to convert to matrices
        :type df: DataFrame
        :return: Generated matrices as NumPy array
        :rtype: numpy.ndarray
        """
        # Simplified implementation for type checking purposes
        self.data = np.array(df)
        self.processed = True
        return cast(NDArray, self.data)


class ProcessedData:
    """
    Encapsulates processed data results for further operations and analysis.

    This class is designed to hold the outcome of data processing procedures.
    It provides a structure to manage and safely access the processed data
    and metadata. The class serves as a foundational component in pipelines
    requiring systematic handling of transformed data.

    :ivar processed_content: Stores the primary content resulting from data
        processing.
    :type processed_content: str
    :ivar timestamp: Records the timestamp when the data was processed.
    :type timestamp: datetime.datetime
    :ivar metadata: Contains additional information or context about the
        processed data.
    :type metadata: dict
    """

    def __init__(
        self, dataframe: DataFrame, matrices: NDArray, tensor_metadata: Dict[str, Any]
    ) -> None:
        self.processed_content: DataFrame = dataframe
        self.matrices: NDArray = matrices
        self.tensor_metadata: Dict[str, Any] = tensor_metadata
        self.timestamp = timestamp.now()


class DataHandler:
    """
    Facilitates data processing by integrating file management, DataFrame
    processing, and matrix generation functionalities.

    A supporting class that orchestrates different components like file
    operations, processing of data into DataFrames, and numerical matrix
    operations. Designed for asynchronous workflows, enabling efficient
    handling of large datasets.

    :ivar df_processor: Handles the creation and manipulation of pandas
        DataFrames.
    :type df_processor: DataFrameProcessor
    :ivar numpy_engine: Manages operations involving numerical matrices
        using numpy.
    :type numpy_engine: NumpyProcessor
    :ivar file_manager: Responsible for file reading and writing operations.
    :type file_manager: FileManager
    """

    df_processor: Optional[DataFrameProcessor] = None
    numpy_engine: Optional[NumpyProcessor] = None
    file_manager: Optional[FileManager] = None

    def __init__(self) -> None:
        self.df_processor: DataFrameProcessor = DataFrameProcessor()
        self.numpy_engine: NumpyProcessor = NumpyProcessor()
        self.file_manager: FileManager = FileManager()

    def _extract_metadata(self, df: DataFrame) -> Dict[str, Any]:
        """
        Extracts metadata from a given DataFrame.

        This method processes the input DataFrame to extract and return
        relevant metadata. Typically used as an internal utility within
        the application to gather specific information from the structure or
        content of the DataFrame.

        :param df: Pandas DataFrame from which metadata will be extracted
        :type df: pandas.DataFrame
        :return: Extracted metadata as a structured object
        :rtype: dict
        """
        return {
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "shape": df.shape,
            "memory_usage": df.memory_usage(deep=True).to_dict(),
            "missing_values": df.isna().sum().to_dict(),
            "null_values": df.isnull().sum().to_dict(),
            "unique_values": df.nunique().to_dict(),
        }

    async def process_data(self, data_source: Union[str, Path]) -> ProcessedData:
        """
        Processes the input data and returns a processed data object containing a
        dataframe, generated matrices, and extracted metadata. The input data
        can be provided as a string or a file path. If a file path is given, the
        function reads the file's contents. The data is then converted into a
        pandas dataframe and processed further to generate computational matrices
        and metadata information.

        :param data_source: The source of the data to be processed. Can be
            provided as a string containing the data or as a file path. If a
            file path is provided, it will be read to obtain the data.
        :type data_source: Union[str, Path]

        :return: A `ProcessedData` object encapsulating the processed dataframe,
            generated matrices, and extracted metadata after processing.
        :rtype: ProcessedData
        """
        # Ensure file_manager is initialized
        if self.file_manager is None:
            self.file_manager = FileManager()

        # Ensure df_processor is initialized
        if self.df_processor is None:
            self.df_processor = DataFrameProcessor()

        # Ensure numpy_engine is initialized
        if self.numpy_engine is None:
            self.numpy_engine = NumpyProcessor()

        if isinstance(data_source, Path):
            data = await self.file_manager.read_file(data_source)
        else:
            data = data_source
            if isinstance(data, str):
                data = ast.literal_eval(data)
                print(data)
                print(type(data))
                print(data[0])
                print(type(data[0]))
                print(data[0]["metadata"])
                print(type(data[0]["metadata"]))
                print(data[0]["metadata"]["columns"])
                print(type(data[0]["metadata"]["columns"]))
                print(data[0]["metadata"]["dtypes"])
                print(type(data[0]["metadata"]["dtypes"]))
                print(data[0]["metadata"]["shape"])
                print(type(data[0]["metadata"]["shape"]))
                print(data[0]["metadata"]["memory_usage"])

        df = self.df_processor.create_dataframe(data)
        matrices = self.numpy_engine.generate_matrices(df)
        print(matrices)
        print(df)
        print(type(df))
        print(df.dtypes)
        print(type(df.dtypes))
        print(df.shape)
        print(type(df.shape))
        print(df.memory_usage(deep=True))
        print(type(df.memory_usage(deep=True)))
        print(df.isna().sum())
        print(type(df.isna().sum()))
        print(df.isnull().sum())
        print(type(df.isnull().sum()))

        return ProcessedData(
            dataframe=df, matrices=matrices, metadata=self._extract_metadata(df)
        )
