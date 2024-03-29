"""Models used across the application.
"""

# Third-party imports
from django.contrib.gis.db.models import MultiPolygonField, PointField
from django.db import models


class TimestampedModel(models.Model):
    """An abstract model used to record creation and update times."""

    created_at_utc = models.DateTimeField(auto_now_add=True)
    last_updated_at_utc = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class FoodwareModel(TimestampedModel):
    """Represents a foodware collection and distribution system for a locale."""

    name = models.CharField()
    boundary = MultiPolygonField(null=True)
    padlet_board_id = models.CharField(blank=True, default="")
    padlet_board_url = models.URLField(blank=True, default="")

    class Meta:
        db_table = "model"
        constraints = [
            models.UniqueConstraint(name="unique_model_name", fields=["name"])
        ]


class FoodwareModelJob(TimestampedModel):
    """Represents a processing job for a foodware model."""

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

    model = models.ForeignKey("FoodwareModel", on_delete=models.CASCADE)
    type = models.CharField(choices=JobType)
    status = models.CharField(choices=JobStatus)
    created_at_utc = models.DateTimeField(auto_now_add=True)
    last_updated_at_utc = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "job"


class FoodwareModelBin(TimestampedModel):
    """Represents a bin used by the model for foodware collection or distribution."""

    class BinPurpose(models.TextChoices):
        """Enumerates purposes for bins."""

        INDOOR = "Indoor Point"
        OUTDOOR = "Outdoor Point"

    class BinLocationCategory(models.TextChoices):
        """Enumerates standardized categories of bin locations."""

        COMMERCIAL_FOODSERVICE = "Commercial Foodservice"
        COMMUNITY_CENTER = "Community Center"
        LIBRARY = "Library"
        LODGING = "Lodging"
        MALLS_AND_SHOPPING_CENTERS = "Mall or Shopping Center"
        MEDICAL_CENTER = "Medical Center"
        OFFICE = "Office Space"
        POST_OFFICE = "Post Office"
        SCHOOL = "School"
        RETAIL_FOOD_AND_DRUG = "Grocery Store, Drugstore, or Pharmacy"
        TRANSPORTATION = "Transportation"
        ZOO_OR_AQUARIUM = "Zoo or Aquarium"

    model = models.ForeignKey("FoodwareModel", on_delete=models.CASCADE)
    external_source = models.CharField()
    external_id = models.CharField(blank=True, default="")
    external_name = models.CharField(blank=True, default="")
    external_category = models.CharField()
    name = models.CharField(blank=True, default="")
    category = models.CharField(choices=BinLocationCategory)
    capacity = models.IntegerField(null=True)
    coords = PointField()
    notes = models.TextField(blank=True, default="")

    class Meta:
        db_table = "location"
        constraints = [
            models.UniqueConstraint(
                name="unique_location",
                fields=["model", "external_source", "external_id"],
            )
        ]
