"""CLI commands for ccutils (Claude Code utilities)."""

import click
from click_default_group import DefaultGroup

from .local import convert_cmd
from .import_cmd import import_cmd
from .open_cmd import open_cmd
from .utils import (
    is_url,
    fetch_url_to_tempfile,
    resolve_credentials,
    format_session_for_display,
)


@click.group(cls=DefaultGroup, default="convert", default_if_no_args=True)
@click.version_option(None, "-v", "--version", package_name="ccutils")
def cli():
    """Convert Claude Code sessions to HTML pages or DuckDB warehouses.

    One command does the converting: no arguments picks sessions
    interactively, PATHS converts the files you name, and --source walks
    everything under a directory.
    """
    pass


# One conversion command (the default), plus the two that are not
# conversion: importing a different source, and opening what was built.
#
# `local`, `all`, `convert`, `web` and `schema` were removed in 0.20.0 with
# no aliases. `local` and `all` were one operation split by scope, and the
# split is what let their behaviour drift -- global post-loop sources ran on
# one path and not the other. `web` and `schema` were deleted outright: one
# read a different data source through an undocumented API, the other was a
# generic JSON introspector, and neither had a path to the warehouse.
cli.add_command(convert_cmd, "convert")
cli.add_command(import_cmd, "import")
cli.add_command(open_cmd, "open")


# Tombstones for the removed names.
#
# Not aliases -- each one exits nonzero and says what to run instead. They
# exist because without them a DefaultGroup FORWARDS an unknown token to the
# default command, so `ccutils local --help` printed the conversion help and
# exited 0: a silent redirect, which is the one thing a hard break must not
# do. `ccutils local` on its own was no better, failing with "File not
# found: local" and leaving the user to guess why.
_REMOVED = {
    "local": "ccutils (no arguments picks sessions; PATHS converts them)",
    "all": "ccutils --source",
    "web": None,
    "schema": None,
}


def _tombstone(name, replacement):
    @click.command(
        name,
        context_settings={"ignore_unknown_options": True,
                          "help_option_names": []},
    )
    @click.argument("ignored", nargs=-1, type=click.UNPROCESSED)
    def _cmd(ignored):
        if replacement:
            raise click.UsageError(
                f"`ccutils {name}` was removed in 0.20.0. Use: {replacement}"
            )
        raise click.UsageError(
            f"`ccutils {name}` was removed in 0.20.0 and has no replacement."
        )

    _cmd.hidden = True
    return _cmd


for _name, _replacement in _REMOVED.items():
    cli.add_command(_tombstone(_name, _replacement), _name)


def main():
    cli()


__all__ = [
    "cli",
    "main",
    "convert_cmd",
    "import_cmd",
    "open_cmd",
    "is_url",
    "fetch_url_to_tempfile",
    "resolve_credentials",
    "format_session_for_display",
]
