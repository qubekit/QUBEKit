import abc
from typing import Any, Dict

from qubekit.utils.datastructures import SchemaBase


class SolventBase(SchemaBase, abc.ABC):
    @abc.abstractmethod
    def format_keywords(self) -> Dict[str, Any]:
        """
        Format the options into a dict that can be consumed by the program harness.
        """
        ...
