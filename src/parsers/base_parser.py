from abc import ABC, abstractmethod

class BaseParser(ABC):
    """
    Abstract base class for all chat parsers.
    Defines the contract that all concrete parsers must follow.
    """
    @abstractmethod
    def parse(self, file_path):
        """
        Parses a given chat file.

        Args:
            file_path (str): The path to the chat file to be parsed.

        Returns:
            list: A list of standardized message dictionaries.
        """
        pass
