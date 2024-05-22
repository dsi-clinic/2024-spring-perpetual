"""Views for the Django application.
"""

# Standard library imports
import json

from django.http import JsonResponse
from django.views import View
from django_q.models import OrmQ
# Third-party imports
from django_q.tasks import Task, async_task
# Application imports
from foodware.models import FoodwareProjectJob


class FoodwareJobView(View):
    """Provides API endpoints to manage foodware jobs."""

    http_method_names = ["get", "post"]

    def post(self, request, *args, **kwargs):
        """Queues a new processing job in the database."""
        # Initialize base error message
        base_err_msg = "Failed to queue processing job."

        # Parse data from request body
        try:
            data = json.loads(request.body.decode("utf-8"))
            project_id = data["project_id"]
            job_type = data["job_type"]
        except json.JSONDecodeError:
            return JsonResponse(
                f"{base_err_msg} Request body could not be parsed as JSON.",
                status=400,
                safe=False,
            )
        except KeyError as e:
            return JsonResponse(
                f'{base_err_msg} Request body missing expected value "{e}".',
                status=400,
                safe=False,
            )

        # Queue asynchronous task
        if job_type == FoodwareProjectJob.JobType.LOCATIONS:
            task_id = async_task(
                "django.core.management.call_command",
                "fetch_locations",
                project_id,
            )
        else:
            return JsonResponse(
                f"{base_err_msg} Received an unexpected job type,"
                f' "{job_type}".',
                status=400,
                safe=False,
            )

        # Return id of task to monitor
        return JsonResponse(task_id, status=201, safe=False)

    def get(self, request, pk):
        """Retrieves the status of a processing job."""
        # Determine if task is currently processing (i.e., within the queues)
        for obj in OrmQ.objects.all():
            if pk == str(obj.task_id()):
                return JsonResponse(
                    {"status": "In Progress", "error": None}, status=200
                )

        # Report terminal status of completed tasks
        task = Task.objects.get(id=pk)
        if task.success:
            payload = {"status": "Success", "error": None}
        else:
            payload = {"status": "Error", "error": task.result}

        return JsonResponse(payload, status=200)
