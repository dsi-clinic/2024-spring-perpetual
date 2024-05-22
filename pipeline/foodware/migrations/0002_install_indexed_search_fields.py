# Manually created

from django.db import migrations


class Migration(migrations.Migration):
    initial = False

    dependencies = [
        ("foodware", "0001_initial"),
    ]

    operations = [
        migrations.RunSQL(
            sql="""
              ALTER TABLE locale
              ADD COLUMN name_vector tsvector
              GENERATED ALWAYS AS (to_tsvector('english', name)) STORED;
            """,
            reverse_sql="""
              ALTER TABLE locale DROP COLUMN name_vector;
            """,
        ),
        migrations.RunSQL(
            sql="""
                CREATE INDEX name_vector_idx
                ON locale
                USING gin(name_vector);
            """,
            reverse_sql="""
                DROP INDEX name_vector_idx
            """,
        ),
    ]
