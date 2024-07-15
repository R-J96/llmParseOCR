import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from torch.utils.data import Dataset
from utils import recur_find_ext, parse_pickle, parse_pickle_full
from google.cloud.documentai_v1.types.document import Document
import pandas as pd


class ReportDataset(Dataset):
    """
    A custom Dataset for loading and processing report data.

    This dataset class is designed to work with pickle files containing report data,
    which can be either Document objects or custom report formats. It supports
    filtering based on a cohort dataframe and provides tokenization functionality.

    Attributes:
        data_dir (Path): Directory containing the report data files.
        tokenizer: Tokenizer object used for encoding text.
        file_list (List[str]): List of file paths to be processed.

    Args:
        data_dir (str): Path to the directory containing report data files.
        tokenizer: Tokenizer object used for encoding text.
        cohort_df (Optional[pd.DataFrame]): DataFrame containing cohort information
            for filtering files. Default is None.
    """

    def __init__(
        self, data_dir: str, tokenizer, cohort_df: Optional[pd.DataFrame] = None
    ):
        self.data_dir: Path = Path(data_dir)
        self.tokenizer = tokenizer
        self.file_list: List[str] = recur_find_ext(data_dir, [".pickle"])

        if cohort_df is not None:
            cohort_list: List[str] = cohort_df["file_names"].tolist()
            self.file_list = [x for x in self.file_list if Path(x).stem in cohort_list]

    def __len__(self) -> int:
        """Returns the number of files in the dataset."""
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, str], int, int, str]:
        """
        Retrieve and process a single item from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            Tuple containing:
                - Dict[str, str]: Parsed pages of the document.
                - int: Maximum token length among all pages.
                - int: Total token length of all pages.
                - str: Stem of the file name.
        """
        file_id: str = self.file_list[idx]
        with open(file_id, "rb") as f:
            doc = pickle.load(f)

        if isinstance(doc, Document):
            pages: Dict[str, str] = parse_pickle_full(doc)
        else:
            pages: Dict[str, str] = parse_pickle(doc)

        stem: str = Path(file_id).stem

        max_length: int = 0
        total_length: int = 0
        for _, page in pages.items():
            page_length = len(self.tokenizer.encode(page))
            max_length = max(max_length, page_length)
            total_length += page_length

        return pages, max_length, total_length, stem
