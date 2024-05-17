"""Models used across the Django project.
"""

# Third-party imports
from django.db import models


class TimestampedModel(models.Model):
    """An abstract model used to record creation and update times."""

    created_at_utc = models.DateTimeField(auto_now_add=True)
    last_updated_at_utc = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True
