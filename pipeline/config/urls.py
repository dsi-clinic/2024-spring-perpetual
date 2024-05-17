from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path
from foodware.views import FoodwareJobView

urlpatterns = [
    path("admin/", admin.site.urls),
    path("foodware/jobs", FoodwareJobView.as_view(), name="foodware-job-list"),
    path(
        "foodware/jobs/<str:pk>",
        FoodwareJobView.as_view(),
        name="foodware-job-detail",
    ),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
