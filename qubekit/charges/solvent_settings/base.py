import abc
from typing import Dict, List

from qubekit.utils.datastructures import SchemaBase


class SolventBase(SchemaBase, abc.ABC):
    @abc.abstractmethod
    def format_keywords(self) -> Dict[str, List[str]]:
        """
        Format the options into a dict that can be consumed by the program harness.
        """
        ...
