from pathlib import Path
from typing import Dict, List, Optional


class LanguagePairMapper:
    """Handles language pair mappings and file naming conventions."""

    def __init__(self, custom_mappings: Optional[Dict[str, str]] = None):
        """
        Initialize with default or custom language pair mappings.

        Args:
            custom_mappings: Optional custom mappings to override defaults.
        """
        self.default_mappings = {
            "en-de": "ende",
            "en-ja": "enja",
            "en-zh": "enzh",
            "en-cs": "encs",
        }
        self.mappings = custom_mappings or self.default_mappings

    def get_prefix(self, language_pair: str) -> str:
        """
        Get the file prefix for a given language pair.

        Args:
            language_pair: Language pair in format "src-tgt" (e.g., "en-de").

        Returns:
            File prefix string.

        Raises:
            ValueError: If language pair is not supported.
        """
        if language_pair not in self.mappings:
            raise ValueError(f"Unsupported language pair: {language_pair}")
        return self.mappings[language_pair]

    def get_supported_pairs(self) -> List[str]:
        """Get list of all supported language pairs."""
        return list(self.mappings.keys())

    def add_language_pair(self, language_pair: str, prefix: str):
        """
        Add a new language pair mapping.

        Args:
            language_pair: Language pair in format "src-tgt".
            prefix: File prefix for this language pair.
        """
        self.mappings[language_pair] = prefix


class DatasetFileLocator:
    """Locates dataset files for specific language pairs and splits."""

    def __init__(self, mapper: LanguagePairMapper):
        """
        Initialize with a language pair mapper.

        Args:
            mapper: LanguagePairMapper instance for prefix resolution.
        """
        self.mapper = mapper

    def get_file_paths(
        self,
        data_dir: str,
        language_pair: str,
        file_template: str = "{prefix}_majority_{split}.tsv",
    ) -> Dict[str, Path]:
        """
        Get file paths for all splits of a language pair.

        Args:
            data_dir: Directory containing the data files.
            language_pair: Language pair to get files for.
            file_template: Template for file naming with {prefix} and {split}
                           placeholders.

        Returns:
            Dictionary mapping split names to file paths.
        """
        prefix = self.mapper.get_prefix(language_pair)
        data_path = Path(data_dir)

        splits = ["train", "dev", "test_blind"]
        file_paths = {}

        for split in splits:
            filename = file_template.format(prefix=prefix, split=split)
            file_path = data_path / filename
            if file_path.exists():
                file_paths[split.replace("_blind", "")] = file_path

        return file_paths

    def find_available_language_pairs(
        self, data_dir: str, file_template: str = "{prefix}_majority_train.tsv"
    ) -> List[str]:
        """
        Find available language pairs based on existing files.

        Args:
            data_dir: Directory to search for data files.
            file_template: Template to check for file existence.

        Returns:
            List of available language pairs.
        """
        data_path = Path(data_dir)
        available_pairs = []

        for language_pair in self.mapper.get_supported_pairs():
            prefix = self.mapper.get_prefix(language_pair)
            filename = file_template.format(prefix=prefix, split="train")
            if (data_path / filename).exists():
                available_pairs.append(language_pair)

        return available_pairs
