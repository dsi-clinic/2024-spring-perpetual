"""Base settings used throughout the Django project.
"""

# Standard library imports
import os
from distutils.util import strtobool
from pathlib import Path

# Third-party imports
from configurations import Configuration


class BaseConfig(Configuration):
    """Defines configuration settings common across environments."""

    # File paths
    BASE_DIR = Path(__file__).parents[3]
    PROJECT_DIR = BASE_DIR / "pipeline"
    DATA_DIR = BASE_DIR / "data"
    POI_DIR = DATA_DIR / "poi"
    CATEGORIES_DIR = DATA_DIR / "categories"
    LOCALES_GEOPARQUET_FPATH = DATA_DIR / "locales.geoparquet"
    CATEGORIES_CROSSWALK_JSON_FPATH = CATEGORIES_DIR / "category_crosswalk.json"
    TEST_BOUNDARIES_DIR = DATA_DIR / "boundaries"
    STATIC_ROOT = os.path.join(PROJECT_DIR, "staticfiles")
    STATIC_URL = "/static/"

    # Default field for primary keys
    DEFAULT_AUTO_FIELD = "django.db.models.AutoField"

    # Installed apps
    INSTALLED_APPS = (
        # Default
        "django.contrib.admin",
        "django.contrib.auth",
        "django.contrib.contenttypes",
        "django.contrib.sessions",
        "django.contrib.messages",
        "django.contrib.staticfiles",
        # Third party apps
        "corsheaders",
        "django_q",
        # Your apps
        "common",
        "foodware",
    )

    # https://django-q.readthedocs.io/en/latest/install.html
    Q_CLUSTER = {
        "name": "DjangORM",
        "workers": 4,
        "timeout": 90,
        "retry": 120,
        "queue_limit": 50,
        "bulk": 10,
        "orm": "default",
        "max_attempts": 1,
    }
    SECRET_KEY = f"{os.getenv('POSTGRES_PASSWORD', '')}://postgres:postgres@postgis:5432/postgres?schema=public"

    # https://docs.djangoproject.com/en/2.0/topics/http/middleware/
    MIDDLEWARE = (
        "corsheaders.middleware.CorsMiddleware",
        "django.middleware.security.SecurityMiddleware",
        "django.contrib.sessions.middleware.SessionMiddleware",
        "django.middleware.common.CommonMiddleware",
        # "django.middleware.csrf.CsrfViewMiddleware",
        "django.contrib.auth.middleware.AuthenticationMiddleware",
        "django.contrib.messages.middleware.MessageMiddleware",
        "django.middleware.clickjacking.XFrameOptionsMiddleware",
    )

    # Email
    EMAIL_BACKEND = "django.core.mail.backends.smtp.EmailBackend"

    # General
    ADMINS = (("Author", ""),)
    LANGUAGE_CODE = "en-us"
    TIME_ZONE = "UTC"

    # If you set this to False, Django will make some optimizations so as not
    # to load the internationalization machinery.
    USE_I18N = False
    USE_L10N = True
    USE_TZ = True

    # URLs
    APPEND_SLASH = False
    ROOT_URLCONF = "config.urls"
    LOGIN_REDIRECT_URL = "/"

    # Set DEBUG to False as a default for safety
    # https://docs.djangoproject.com/en/dev/ref/settings/#debug
    DEBUG = strtobool(os.getenv("DJANGO_DEBUG", "no"))

    # Password Validation
    # https://docs.djangoproject.com/en/2.0/topics/auth/passwords/#module-django.contrib.auth.password_validation
    AUTH_PASSWORD_VALIDATORS = [
        {
            "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
        },
        {
            "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
        },
        {
            "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
        },
        {
            "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
        },
    ]

    # Templates
    TEMPLATES = [
        {
            "BACKEND": "django.template.backends.django.DjangoTemplates",
            "DIRS": [],
            "APP_DIRS": True,
            "OPTIONS": {
                "context_processors": [
                    "django.template.context_processors.debug",
                    "django.template.context_processors.request",
                    "django.contrib.auth.context_processors.auth",
                    "django.contrib.messages.context_processors.messages",
                ],
            },
        },
    ]

    # Database
    # https://docs.djangoproject.com/en/3.2/ref/settings/#databases
    DATABASES = {
        "default": {
            "ENGINE": "django.contrib.gis.db.backends.postgis",
            "NAME": os.getenv("POSTGRES_DB", "postgres"),
            "USER": os.getenv("POSTGRES_USER", "postgres"),
            "PASSWORD": os.getenv("POSTGRES_PASSWORD", ""),
            "HOST": os.getenv("POSTGRES_HOST", "postgres"),
            "PORT": int(os.getenv("POSTGRES_PORT", 5432)),
            "CONN_MAX_AGE": int(os.getenv("POSTGRES_CONN_MAX_AGE", 0)),
            "DISABLE_SERVER_SIDE_CURSORS": False,
            "OPTIONS": {"sslmode": "require"},
        }
    }

    # Caching
    CACHES = {
        "default": {
            "BACKEND": "django.core.cache.backends.db.DatabaseCache",
            "LOCATION": "default_cache",
        }
    }

    # Logging
    LOGGING = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "django.server": {
                "()": "django.utils.log.ServerFormatter",
                "format": "[%(server_time)s] %(message)s",
            },
            "verbose": {
                "format": "%(levelname)s %(asctime)s %(module)s %(process)d %(thread)d %(message)s"
            },
            "simple": {"format": "%(levelname)s %(message)s"},
        },
        "filters": {
            "require_debug_true": {
                "()": "django.utils.log.RequireDebugTrue",
            },
        },
        "handlers": {
            "django.server": {
                "level": "INFO",
                "class": "logging.StreamHandler",
                "formatter": "django.server",
            },
            "console": {
                "level": "DEBUG",
                "class": "logging.StreamHandler",
                "formatter": "simple",
            },
            "mail_admins": {
                "level": "ERROR",
                "class": "django.utils.log.AdminEmailHandler",
            },
        },
        "loggers": {
            "django": {
                "handlers": ["console"],
                "propagate": True,
            },
            "django.server": {
                "handlers": ["django.server"],
                "level": "INFO",
                "propagate": False,
            },
            "django.request": {
                "handlers": ["mail_admins", "console"],
                "level": "ERROR",
                "propagate": False,
            },
            "django.db.backends": {"handlers": ["console"], "level": "INFO"},
        },
    }
