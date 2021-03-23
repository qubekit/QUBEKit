from typing import Dict, Generator, List, Union

from pydantic import BaseModel, Field

from QUBEKit.forcefield.parameters import VirtualSite3Point, VirtualSite4Point

SiteTypes = Union[VirtualSite3Point, VirtualSite4Point]


class VirtualSiteGroup(BaseModel):
    """
    A basic class to organise virtual sites.
    """

    sites: Dict[int, List[SiteTypes]] = Field(
        {},
        description="The virtual site parameters stored under the index of the parent atom.",
    )

    @property
    def n_sites(self) -> int:
        return sum([len(values) for values in self.sites.values()])

    def iter_sites(self) -> Generator[SiteTypes, None, None]:
        """A helpful way to iterate over all stored sites."""
        for sites in self.sites.values():
            for site in sites:
                yield site

    def create_site(self, parent_index: int, **kwargs) -> int:
        """Create a new virtual site for the parent atom and add it to the goup.

        Args:
            parent_index: The index of the parent atom.
            kwargs: The keywords needed to create the site.

        Returns:
            The index of the parent atom the site was added to.
        """
        if "closest_c_index" in kwargs:
            site = VirtualSite4Point(parent_index, **kwargs)
        else:
            site = VirtualSite3Point(parent_index, **kwargs)

        self.sites.setdefault(parent_index, []).append(site)
        return parent_index

    def add_site(self, site: SiteTypes) -> int:
        """
        Add a virtual site object to the group.
        """

        self.sites.setdefault(site.parent_index, []).append(site)
        return site.parent_index

    def get_sites(self, parent_index: int) -> List[SiteTypes]:
        """
        Get a list of sites for the given parent atom index.
        """
        sites = self.sites.get(parent_index, [])
        return sites

    def remove_sites(self, parent_index: int) -> List[SiteTypes]:
        """Return the list of sites removed from the parent index."""
        if parent_index in self.sites:
            sites = self.sites.pop(parent_index)
        else:
            sites = []
        return sites
