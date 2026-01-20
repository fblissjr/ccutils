"""Schema inspection command for analyzing JSON structure."""

import json
from pathlib import Path

import click

from ..parsers.schema_inspector import (
    format_schema,
    inspect_export_directory,
    inspect_json_file,
)


@click.command("schema")
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--samples",
    "-s",
    default=5,
    help="Number of array items to sample (default: 5).",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output raw JSON schema instead of formatted text.",
)
@click.option(
    "--file",
    "-f",
    "single_file",
    help="Inspect only this file within a directory.",
)
def schema_cmd(path, samples, output_json, single_file):
    """Inspect JSON structure without exposing content.

    Analyzes JSON files to show their schema (keys, types, array lengths)
    without revealing any actual values. Output is safe to share publicly
    or paste into AI assistants.

    PATH can be a single JSON file or a directory containing JSON files.

    Examples:

    \b
      # Inspect a single file
      ccutils schema conversations.json

    \b
      # Inspect all files in an export directory
      ccutils schema ./my-claude-export/

    \b
      # Output as JSON for programmatic use
      ccutils schema ./export --json > schema.json

    \b
      # Inspect specific file in a directory
      ccutils schema ./export -f conversations.json
    """
    path = Path(path)

    if path.is_file():
        # Single file
        result = inspect_json_file(path, max_array_samples=samples)
        _output_result(result, output_json)

    elif path.is_dir():
        if single_file:
            # Specific file in directory
            target = path / single_file
            if not target.exists():
                raise click.ClickException(f"File not found: {target}")
            result = inspect_json_file(target, max_array_samples=samples)
            _output_result(result, output_json)
        else:
            # All JSON files in directory
            results = inspect_export_directory(path, max_array_samples=samples)
            if not results:
                raise click.ClickException(f"No JSON files found in {path}")

            if output_json:
                click.echo(json.dumps(results, indent=2, default=str))
            else:
                for filename, result in results.items():
                    click.echo(f"\n{'=' * 60}")
                    click.echo(f"FILE: {filename}")
                    click.echo("=" * 60)
                    if "error" in result:
                        click.echo(f"  Error: {result['error']}")
                    else:
                        _output_result(result, output_json=False)
    else:
        raise click.ClickException(f"Path is neither file nor directory: {path}")


def _output_result(result: dict, output_json: bool):
    """Output a single file inspection result."""
    if output_json:
        click.echo(json.dumps(result, indent=2, default=str))
        return

    # Check for error
    if "error" in result:
        click.echo(f"Error: {result['error']}")
        return

    # Formatted text output
    size_kb = result["size_bytes"] / 1024
    file_format = result.get("format", "json")
    click.echo(f"Format: {file_format.upper()}")
    click.echo(f"Size: {size_kb:.1f} KB")

    if "line_count" in result:
        click.echo(f"Lines: {result['line_count']}")

    click.echo()
    click.echo(format_schema(result["schema"]))
