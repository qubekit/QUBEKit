import click

import qubekit
from qubekit.cli.bulk import bulk
from qubekit.cli.combine import combine
from qubekit.cli.config import config
from qubekit.cli.progress import progress
from qubekit.cli.run import restart, run


@click.group()
@click.version_option(version=qubekit.__version__, prog_name="QUBEKit")
def cli():
    pass


cli.add_command(config)
cli.add_command(run)
cli.add_command(restart)
cli.add_command(bulk)
cli.add_command(progress)
cli.add_command(combine)

if __name__ == "__main__":
    cli()
