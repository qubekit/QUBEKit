import abc
from typing import Dict, List

from qubekit.utils.datastructures import QCOptions


class SolventBase(QCOptions, abc.ABC):
    @abc.abstractmethod
    def format_keywords(self) -> Dict[str, List[str]]:
        """
        Format the options into a dict that can be consumed by the gaussian harness.
        """
        ...
