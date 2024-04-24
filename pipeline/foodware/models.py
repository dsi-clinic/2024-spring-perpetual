"""Models used across the application.
"""

# Application imports
from common.models import TimestampedModel

# Third-party imports
from django.contrib.gis.db.models import MultiPolygonField, PointField
from django.db import models


class Locale(models.Model):
    """Represents a potential locale for a foodware project."""

    class Meta:
        db_table = "locale"
        constraints = [
            models.UniqueConstraint(
                name="unique_locale",
                fields=["name"],
            )
        ]

    name = models.CharField()
    geometry = MultiPolygonField()


class FoodwareProject(TimestampedModel):
    """Represents a foodware collection and distribution project."""

    class Meta:
        db_table = "foodware_project"

    name = models.CharField()
    description = models.TextField(null=True)
    locale = models.ForeignKey("Locale", on_delete=models.CASCADE)


class FoodwareProjectJob(TimestampedModel):
    """Represents a processing job for a foodware project."""

    class JobType(models.TextChoices):
        """Enumerates the type of job workflow."""

        LOCATIONS = "Location"
        ROUTES = "Route"

    class JobStatus(models.TextChoices):
        """Enumerates potential statuses for a job."""

        NOT_STARTED = "Not Started"
        IN_PROGRESS = "In Progress"
        ERROR = "Error"
        SUCCESS = "Success"

    model = models.ForeignKey("FoodwareProject", on_delete=models.CASCADE)
    type = models.CharField(choices=JobType)
    status = models.CharField(choices=JobStatus)
    last_error = models.TextField()

    class Meta:
        db_table = "foodware_project_job"


class FoodwareProjectBinCandidate(TimestampedModel):
    """Represents a potential bin location for foodware collection or distribution."""

    class BinClassification(models.TextChoices):
        """Enumerates bin types."""

        INDOOR = "Indoor"
        OUTDOOR = "Outdoor"
        EXCLUDED = "Excluded"

    model = models.ForeignKey("FoodwareProject", on_delete=models.CASCADE)
    external_source = models.ForeignKey("PoiSource", on_delete=models.CASCADE)
    external_id = models.CharField(blank=True, default="")
    name = models.CharField(blank=True, default="")
    classification = models.CharField(choices=BinClassification)
    external_category = models.ForeignKey("PoiCategory", on_delete=models.CASCADE)
    mapped_category = models.ForeignKey("PoiCategory", on_delete=models.CASCADE)
    formatted_address = models.CharField(blank=True, default="")
    street = models.CharField(blank=True, default="")
    city = models.CharField(blank=True, default="")
    state = models.CharField(blank=True, default="")
    zipcode = models.CharField(blank=True, default="")
    coords = PointField()

    class Meta:
        db_table = "foodware_project_bin"
        constraints = [
            models.UniqueConstraint(
                name="unique_foodware_project_bin",
                fields=[
                    "model",
                    "external_source",
                    "external_id",
                    "name",
                    "formatted_address",
                ],
            )
        ]
