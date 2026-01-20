"""Privacy-safe JSON schema inspector.

Analyzes JSON structure without exposing actual content values.
Useful for understanding export file formats without loading
sensitive data into context or sharing personal information.

Design principles:
- Never extract string values (only lengths and patterns)
- Sample arrays to understand structure, report counts
- Redact by default - assume all leaf values are sensitive
- Output is shareable - contains zero personal data
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Optional


def infer_type(value: Any) -> str:
    """Get the type name for a value."""
    if value is None:
        return "null"
    elif isinstance(value, bool):
        return "boolean"
    elif isinstance(value, int):
        return "integer"
    elif isinstance(value, float):
        return "number"
    elif isinstance(value, str):
        return "string"
    elif isinstance(value, list):
        return "array"
    elif isinstance(value, dict):
        return "object"
    else:
        return type(value).__name__


def classify_string(value: str) -> dict:
    """Classify a string by pattern without exposing content.

    Returns metadata about the string type, not the value itself.
    """
    length = len(value)

    # Detect common patterns
    if re.match(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", value, re.I
    ):
        return {"_type": "string", "_format": "uuid", "_length": length}

    if re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", value):
        return {"_type": "string", "_format": "iso8601", "_length": length}

    if re.match(r"^\d{4}-\d{2}-\d{2}$", value):
        return {"_type": "string", "_format": "date", "_length": length}

    if value in ("true", "false"):
        return {"_type": "string", "_format": "boolean_string", "_length": length}

    if re.match(r"^https?://", value):
        return {"_type": "string", "_format": "url", "_length": length}

    if re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", value):
        return {"_type": "string", "_format": "email", "_length": length}

    # Check if it looks like an enum (short, no spaces, limited charset)
    if length <= 30 and re.match(r"^[a-z_]+$", value):
        return {
            "_type": "string",
            "_format": "enum_like",
            "_length": length,
            "_example_pattern": value,
        }

    # Generic string
    return {"_type": "string", "_length": length}


def merge_string_schemas(schemas: list[dict]) -> dict:
    """Merge multiple string schemas to find common patterns."""
    if not schemas:
        return {"_type": "string"}

    formats = Counter(s.get("_format") for s in schemas)
    lengths = [s.get("_length", 0) for s in schemas]

    result = {
        "_type": "string",
        "_length_min": min(lengths),
        "_length_max": max(lengths),
    }

    # If all have same format, report it
    if len(formats) == 1:
        fmt = list(formats.keys())[0]
        if fmt:
            result["_format"] = fmt
    elif formats:
        # Report format distribution
        result["_formats"] = dict(formats)

    # Collect enum-like values
    enum_values = [
        s.get("_example_pattern")
        for s in schemas
        if s.get("_format") == "enum_like" and s.get("_example_pattern")
    ]
    if enum_values:
        unique_values = [v for v in set(enum_values) if v is not None]
        if len(unique_values) <= 10:
            result["_enum_values"] = sorted(unique_values)

    return result


def infer_schema(
    obj: Any,
    max_array_samples: int = 5,
    path: str = "$",
) -> dict:
    """Recursively infer schema from a JSON object.

    Args:
        obj: The JSON object to analyze
        max_array_samples: Maximum array items to sample for schema inference
        path: Current path in the object (for debugging)

    Returns:
        A schema dict describing the structure without sensitive values
    """
    if obj is None:
        return {"_type": "null"}

    if isinstance(obj, bool):
        return {"_type": "boolean"}

    if isinstance(obj, int):
        return {"_type": "integer", "_example_range": (obj, obj)}

    if isinstance(obj, float):
        return {"_type": "number"}

    if isinstance(obj, str):
        return classify_string(obj)

    if isinstance(obj, list):
        schema = {
            "_type": "array",
            "_length": len(obj),
        }

        if not obj:
            schema["_items"] = None
            return schema

        # Sample items to infer item schema
        samples = obj[:max_array_samples]
        item_schemas = [
            infer_schema(item, max_array_samples, f"{path}[]") for item in samples
        ]

        # Check if all items have same type
        item_types = [s.get("_type") for s in item_schemas]
        if len(set(item_types)) == 1:
            # Homogeneous array - merge schemas
            if item_types[0] == "object":
                schema["_items"] = merge_object_schemas(item_schemas)
            elif item_types[0] == "string":
                schema["_items"] = merge_string_schemas(item_schemas)
            else:
                schema["_items"] = item_schemas[0]
        else:
            # Heterogeneous array - report type distribution
            type_counts = Counter(item_types)
            schema["_item_types"] = dict(type_counts)
            schema["_items"] = item_schemas[0]  # Sample first item

        return schema

    if isinstance(obj, dict):
        schema = {"_type": "object", "_keys": {}}

        for key, value in obj.items():
            schema["_keys"][key] = infer_schema(
                value, max_array_samples, f"{path}.{key}"
            )

        return schema

    return {"_type": infer_type(obj)}


def merge_object_schemas(schemas: list[dict]) -> dict:
    """Merge multiple object schemas to find common structure."""
    if not schemas:
        return {"_type": "object", "_keys": {}}

    # Collect all keys and their schemas
    all_keys = {}
    key_counts = Counter()

    for schema in schemas:
        keys = schema.get("_keys", {})
        for key, key_schema in keys.items():
            key_counts[key] += 1
            if key not in all_keys:
                all_keys[key] = []
            all_keys[key].append(key_schema)

    # Merge schemas for each key
    merged_keys = {}
    total = len(schemas)

    for key, key_schemas in all_keys.items():
        presence = key_counts[key] / total

        # Merge the schemas for this key
        if len(key_schemas) == 1:
            merged = key_schemas[0]
        else:
            # Check if all same type
            types = [s.get("_type") for s in key_schemas]
            if len(set(types)) == 1:
                if types[0] == "string":
                    merged = merge_string_schemas(key_schemas)
                elif types[0] == "object":
                    merged = merge_object_schemas(key_schemas)
                else:
                    merged = key_schemas[0]
            else:
                merged = {"_type": "mixed", "_types": dict(Counter(types))}

        # Add presence info if not always present
        if presence < 1.0:
            merged["_presence"] = f"{presence:.0%}"

        merged_keys[key] = merged

    return {"_type": "object", "_keys": merged_keys}


def format_schema(schema: dict, indent: int = 0, key_name: str = "") -> str:
    """Format a schema dict as human-readable text.

    Args:
        schema: The schema dict to format
        indent: Current indentation level
        key_name: The key name if this is an object property

    Returns:
        Formatted string representation
    """
    prefix = "  " * indent
    type_name = schema.get("_type", "unknown")

    # Build the line
    if key_name:
        line = f"{prefix}{key_name}: {type_name}"
    else:
        line = f"{prefix}{type_name}"

    # Add type-specific annotations
    if type_name == "string":
        annotations = []
        if "_format" in schema:
            annotations.append(schema["_format"])
        if "_length_min" in schema and "_length_max" in schema:
            if schema["_length_min"] == schema["_length_max"]:
                annotations.append(f"{schema['_length_min']} chars")
            else:
                annotations.append(
                    f"{schema['_length_min']}-{schema['_length_max']} chars"
                )
        elif "_length" in schema:
            annotations.append(f"{schema['_length']} chars")
        if "_enum_values" in schema:
            vals = schema["_enum_values"]
            if len(vals) <= 5:
                annotations.append(f"values: {vals}")
            else:
                annotations.append(f"{len(vals)} unique values")
        if annotations:
            line += f" ({', '.join(annotations)})"

    elif type_name == "array":
        line += f" [{schema.get('_length', '?')} items]"

    elif type_name == "integer":
        if "_example_range" in schema:
            r = schema["_example_range"]
            if r[0] == r[1]:
                line += f" (e.g. {r[0]})"

    # Add presence indicator
    if "_presence" in schema:
        line += f" (optional, {schema['_presence']})"

    lines = [line]

    # Recurse for nested structures
    if type_name == "object" and "_keys" in schema:
        for key, key_schema in schema["_keys"].items():
            lines.append(format_schema(key_schema, indent + 1, key))

    elif type_name == "array" and "_items" in schema and schema["_items"]:
        items_schema = schema["_items"]
        if items_schema.get("_type") == "object":
            lines.append(f"{prefix}  []: object")
            for key, key_schema in items_schema.get("_keys", {}).items():
                lines.append(format_schema(key_schema, indent + 2, key))
        else:
            lines.append(format_schema(items_schema, indent + 1, "[]"))

    return "\n".join(lines)


def inspect_json_file(
    file_path: Path,
    max_array_samples: int = 5,
) -> dict:
    """Inspect a JSON or JSONL file and return its schema.

    Args:
        file_path: Path to the JSON/JSONL file
        max_array_samples: Max array items to sample

    Returns:
        Dict with schema and file metadata
    """
    file_path = Path(file_path)

    # Check if JSONL (newline-delimited JSON)
    is_jsonl = file_path.suffix.lower() == ".jsonl"

    with open(file_path, "r", encoding="utf-8") as f:
        if is_jsonl:
            # Parse JSONL - collect items into array for schema inference
            items = []
            line_count = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                    line_count += 1
                    # Only sample first N items for schema inference
                    if line_count >= max_array_samples * 2:
                        break
                except json.JSONDecodeError:
                    continue

            if not items:
                return {
                    "file": file_path.name,
                    "size_bytes": file_path.stat().st_size,
                    "format": "jsonl",
                    "error": "No valid JSON lines found",
                }

            # Count total lines for metadata
            f.seek(0)
            total_lines = sum(1 for line in f if line.strip())

            data = items
            schema = infer_schema(data, max_array_samples)

            return {
                "file": file_path.name,
                "size_bytes": file_path.stat().st_size,
                "format": "jsonl",
                "line_count": total_lines,
                "schema": schema,
            }
        else:
            data = json.load(f)

    schema = infer_schema(data, max_array_samples)

    return {
        "file": file_path.name,
        "size_bytes": file_path.stat().st_size,
        "format": "json",
        "schema": schema,
    }


def inspect_export_directory(
    export_path: Path,
    max_array_samples: int = 5,
) -> dict:
    """Inspect all JSON files in a Claude.ai export directory.

    Args:
        export_path: Path to the export directory
        max_array_samples: Max array items to sample

    Returns:
        Dict mapping filenames to their schemas
    """
    export_path = Path(export_path)
    results = {}

    for json_file in sorted(export_path.glob("*.json")):
        try:
            results[json_file.name] = inspect_json_file(json_file, max_array_samples)
        except (json.JSONDecodeError, IOError) as e:
            results[json_file.name] = {"error": str(e)}

    return results
