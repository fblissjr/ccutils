"""Single source for the lineage/parser contract version.

Stamped into dim_etl_version, every fact's created/last_updated version
keys, fact_etl_runs / fact_etl_batch_runs, and the Parquet lake's
parser_version column. MUST be bumped alongside pyproject.toml's version
-- it is how lineage distinguishes rows written under different contracts
(e.g. pre- vs post-subagent-identity-fix). Lives in its own module so
both parsers/ and etl/ can import it without layering either on the
other.
"""

PARSER_VERSION = "0.19.0"
