# src/data/schema_service.py

from sqlalchemy import inspect

# PostgreSQL connection engine
from src.data.db import engine


def get_database_schema():
    """
    PostgreSQL se tables, columns aur foreign keys read karta hai.
    Ye schema later LLM ko diya jayega so that it generates correct SQL.
    """

    # Inspector database structure ko read karta hai
    inspector = inspect(engine)

    schema_lines = []

    # Public schema ke saare tables get karo
    table_names = inspector.get_table_names(schema="public")

    for table_name in table_names:

        schema_lines.append(f"\nTABLE: {table_name}")

        # ------------------------------------------
        # Get columns
        # ------------------------------------------

        columns = inspector.get_columns(
            table_name,
            schema="public"
        )

        for column in columns:

            column_name = column["name"]
            column_type = column["type"]

            schema_lines.append(
                f"  - {column_name}: {column_type}"
            )

        # ------------------------------------------
        # Get foreign-key relationships
        # ------------------------------------------

        foreign_keys = inspector.get_foreign_keys(
            table_name,
            schema="public"
        )

        for fk in foreign_keys:

            source_columns = fk["constrained_columns"]
            target_table = fk["referred_table"]
            target_columns = fk["referred_columns"]

            schema_lines.append(
                f"  FK: {source_columns} "
                f"-> {target_table}.{target_columns}"
            )

    # Convert list into one big string
    return "\n".join(schema_lines)


# Run only when this file is executed directly
if __name__ == "__main__":

    schema = get_database_schema()

    print(schema)