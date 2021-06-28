from typing import Dict, Generator, List, Union

from pydantic import BaseModel, Field

from qubekit.forcefield.parameters import VirtualSite3Point, VirtualSite4Point

SiteTypes = Union[VirtualSite3Point, VirtualSite4Point]


class VirtualSiteGroup(BaseModel):
    """
    A basic class to organise virtual sites.
    """

    sites: Dict[int, List[SiteTypes]] = Field(
        {},
        description="The virtual site parameters stored under the index of the parent atom.",
    )

    def __iter__(self) -> Generator[SiteTypes, None, None]:
        """A helpful way to iterate over all stored sites."""
        for sites in self.sites.values():
            for site in sites:
                yield site

    def __getitem__(self, item):
        return self.sites[item]

    @property
    def n_sites(self) -> int:
        return sum([len(values) for values in self.sites.values()])

    def create_site(self, **kwargs):
        """Create a new virtual site for the parent atom and add it to the group.

        Args:
            kwargs: The keywords needed to create the site.

        Returns:
            The index of the parent atom the site was added to.
        """
        if "closest_c_index" in kwargs:
            site = VirtualSite4Point(**kwargs)
        else:
            site = VirtualSite3Point(**kwargs)

        parent_index = kwargs["parent_index"]
        self.sites.setdefault(parent_index, []).append(site)

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

    def remove_sites(self, parent_index: int):
        """Return the list of sites removed from the parent index."""
        self.sites.pop(parent_index)

    def clear_sites(self) -> None:
        """Remove all sites from the molecule."""
        self.sites = {}
