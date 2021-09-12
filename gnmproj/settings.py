import os
from pathlib import Path
import subprocess

from dotenv import load_dotenv
import sentry_sdk
from sentry_sdk.integrations.celery import CeleryIntegration
from sentry_sdk.integrations.django import DjangoIntegration

# pre-config

load_dotenv()

process = subprocess.Popen(
    ['git', 'describe', '--tags', '--always'],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

stdout, _ = process.communicate()

RELEASE = stdout.decode('utf-8').strip()

# config

ADMINS = [
    ('Denis', 'omelched@gmail.com'),
]

ALLOWED_HOSTS = ['*']

APPEND_SLASH = False

CACHES = {
    'default': {
        'BACKEND': 'gnmproj.apps.gnm.HashedDatabaseCache',
        'LOCATION': 'hashcache_table',
    }
}

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

DATABASES = {
    'default': {
        'ENGINE': os.environ.get('DATABASE_ENGINE', 'django.db.backends.sqlite3'),
        'NAME': os.environ.get('DATABASE_NAME', 'db.sqlite3'),
        'USER': os.environ.get('DATABASE_USER', ''),
        'PASSWORD': os.environ.get('DATABASE_PASSWORD', ''),
        'HOST': os.environ.get('DATABASE_HOST', ''),
        'PORT': os.environ.get('DATABASE_PORT', ''),
        'OPTIONS': {
            'application_name': os.environ.get('DATABASE_application_name', f'gnmproj@{RELEASE}')
        } if os.environ.get('DATABASE_ENGINE', None) == 'django.db.backends.postgresql' else {}
    }
}

DEBUG = os.environ.get('DEBUG', False)

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.environ.get('SECRET_KEY', 'SUPERSECRET!!!')

INSTALLED_APPS = [
    'gnmproj.apps.gnm',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.staticfiles',
    'django.contrib.messages',
    "rest_framework",
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'gnmproj.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': ['templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ]
        },
    }
]

# WSGI_APPLICATION = 'gnmproj.wsgi.application'

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'
    },
]

LANGUAGE_CODE = 'ru-RU'

TIME_ZONE = 'Europe/Moscow'

USE_I18N = True

USE_L10N = True

USE_TZ = True

STATIC_URL = '/static/'

STATIC_ROOT = os.environ.get('STATIC_ROOT', '/var/www/static/')

# Third party services

REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": (
        "rest_framework.authentication.BasicAuthentication",
    ),
}

if not os.environ.get('DISABLE_SENTRY', False):
    sentry_sdk.init(
        dsn=os.environ.get('SENTRY_DSN', None),
        release=RELEASE,
        environment=os.environ.get('SENTRY_ENVIRONMENT', 'NO-ENV'),
        integrations=[
            DjangoIntegration(
                transaction_style='function_name',
            ),
            CeleryIntegration()
        ],
        traces_sample_rate=float(os.environ.get('SENTRY_SAMPLE_RATE', 0)),
        ignore_errors=[
            'Http404'
        ]
    )

CELERY_ACCEPT_CONTENT = ['application/json']
CELERY_RESULT_ACCEPT_CONTENT = ['application/json']
CELERY_TIMEZONE = 'Europe/Moscow'
CELERY_TASK_TRACK_STARTED = True
CELERY_RESULT_BACKEND = 'rpc://'
CELERY_BROKER_URL = (
    f'amqp://'
    f"{os.environ.get('RMQ_USER', 'guest')}:"
    f"{os.environ.get('RMQ_PASS', 'guest')}@"
    f"{os.environ.get('RMQ_HOST', 'localhost')}:"
    f"{os.environ.get('RMQ_PORT', '5672')}/"
    f"{os.environ.get('RMQ_VHOST', '')}"
)
CELERY_IMPORTS = (
    ...,
)
