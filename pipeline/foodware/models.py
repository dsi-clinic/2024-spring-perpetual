"""Models used across the application.
"""

# Application imports
from common.models import TimestampedModel

# Third-party imports
from django.contrib.gis.db.models import MultiPolygonField, PointField
from django.db import models


class PoiProvider(models.Model):
    """Represents a points of interest (POI) provider."""

    name = models.CharField()

    class Meta:
        db_table = "poi_provider"
        constraints = [
            models.UniqueConstraint(
                name="unique_poi_provider",
                fields=["name"],
            )
        ]


class PoiParentCategory(models.Model):
    """Represents a top-level points of interest (POI) category."""

    class Name(models.TextChoices):
        """Enumerates valid category names."""

        ATTRACTIONS = "Attractions"
        EDUCATION = "Education"
        LODGING = "Lodging"
        MEDICAL = "Medical"
        PUBLIC_SERVICES = "Public Services"
        RESTAURANTS = "Restaurants"
        SHOPPING = "Shopping"
        TRANSPORTATION = "Transportation"
        WORKPLACES = "Workplaces"

    name = models.CharField()

    class Meta:
        db_table = "poi_parent_category"
        constraints = [
            models.UniqueConstraint(
                name="unique_poi_parent_category",
                fields=["name"],
            )
        ]


class PoiProviderCategory(models.Model):
    """Represents a category used by a points of interest (POI) source."""

    parent = models.ForeignKey("PoiParentCategory", on_delete=models.CASCADE)
    provider = models.ForeignKey("PoiProvider", on_delete=models.CASCADE)
    name = models.CharField()

    class Meta:
        db_table = "poi_provider_category"
        constraints = [
            models.UniqueConstraint(
                name="unique_poi_provider_category",
                fields=["provider", "name"],
            ),
        ]


class Locale(models.Model):
    """Represents a potential locale for a foodware project."""

    name = models.CharField()
    geometry = MultiPolygonField()

    class Meta:
        db_table = "locale"
        constraints = [
            models.UniqueConstraint(
                name="unique_locale",
                fields=["name"],
            )
        ]


class FoodwareProject(TimestampedModel):
    """Represents a foodware collection and distribution project."""

    name = models.CharField()
    description = models.TextField(null=True)
    locale = models.ForeignKey("Locale", on_delete=models.CASCADE)

    class Meta:
        db_table = "foodware_project"


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

    project = models.ForeignKey("FoodwareProject", on_delete=models.CASCADE)
    type = models.CharField(choices=JobType)
    status = models.CharField(choices=JobStatus)
    last_error = models.JSONField()

    class Meta:
        db_table = "foodware_project_job"


class FoodwareProjectBin(TimestampedModel):
    """Represents a bin location for foodware collection or distribution."""

    class Classification(models.TextChoices):
        """Enumerates bin types."""

        UNDETERMINED = "Undetermined"
        INDOOR = "Indoor"
        OUTDOOR = "Outdoor"
        EXCLUDED = "Excluded"

    project = models.ForeignKey("FoodwareProject", on_delete=models.CASCADE)
    provider = models.ForeignKey("PoiProvider", on_delete=models.CASCADE)
    parent_category = models.ForeignKey("PoiParentCategory", on_delete=models.CASCADE)
    external_id = models.CharField(blank=True, default="")
    classification = models.CharField(choices=Classification)
    name = models.CharField(blank=True, default="")
    external_categories = models.JSONField()
    formatted_address = models.CharField(blank=True, default="")
    coords = PointField()
    notes = models.TextField(blank=True, default="")

    class Meta:
        db_table = "foodware_project_bin"
        constraints = [
            models.UniqueConstraint(
                name="unique_foodware_project_bin",
                fields=[
                    "project",
                    "provider",
                    "external_id",
                    "name",
                    "formatted_address",
                ],
            )
        ]


class PoiCache(TimestampedModel):
    """Temporary cache for POI-related data."""

    project = models.ForeignKey("FoodwareProject", on_delete=models.CASCADE)
    provider = models.ForeignKey("PoiProvider", on_delete=models.CASCADE)
    parent_category = models.ForeignKey("PoiParentCategory", on_delete=models.CASCADE)
    data = models.JSONField()

    class Meta:
        db_table = "poi_cache"
        constraints = [
            models.UniqueConstraint(
                name="unique_poi_cache_item",
                fields=["project", "provider", "parent_category"],
            )
        ]
