# MeetUp 224 - Beginners Python and Machine Learning - Wed 15 Oct 2025 - Django and Strava Auth with PyCharm

## Part 1 - 15 Oct 2025 - Writing a Django Web App and authenticating with Strava 

Links:

- Youtube: <https://youtu.be/_ieZpyktbho>
- Github:  <https://github.com/timcu/bpaml-sessions/blob/master/online/meetup224_tim_django_strava_auth.md>
- Meetup:  <https://www.meetup.com/beginners-python-machine-learning/events/311313362/>

## References

- <https://github.com/timcu/bpaml-sessions/blob/master/online/meetup206_tim_django.md>
- <https://github.com/timcu/bpaml-sessions/blob/master/online/meetup220_tim_strava_api.py>
- <https://www.jetbrains.com/pycharm/>
- <https://docs.djangoproject.com/en/5.2/intro/tutorial01/>
- <https://https://docs.allauth.org/en/latest/>
- <https://developers.strava.com/>
- <https://realpython.com/django-nginx-gunicorn/>

## Procedure

### Create project and app <https://docs.djangoproject.com/en/5.2/intro/tutorial01/>

If you are checking out the source code from GitHub

- `git clone https://github.com/timcu/django_bpaml_strava_site`
- `git checkout step01`

If you are following the script manually

- Create a new PyCharm project `django_bpaml_strava_site`
  - Make sure you use a new virtual environment either venv, conda depending on your python installation
- Open Terminal window
- `pip install django-allauth[socialaccount]` This will also install Django, allauth with Google Auth and other dependencies
- `django-admin startproject django_bpaml_strava_site`  # instead of `mysite` in tutorial
- PyCharm project and Django both have concept of "Project". We need to move all files from django project into pycharm project (up one directory)
  - Move `manage.py` up a folder
  - Move five files `__init__.py`, `asgi.py`, `settings.py`, `urls.py` and `wsgi.py` up a folder
  - Delete empty folder `django_bpaml_strava_site`
- `pip freeze > requirements.txt`
- `python manage.py runserver`
- `python manage.py startapp django_bpaml_strava`

#### Create settings for our app `git checkout step02`

Edit `django_bpaml_strava_site/settings.py` to include our app, and set up where our templates are going to be.

```python
INSTALLED_APPS = [
    'django_bpaml_strava.apps.DjangoBpamlStravaConfig',
    # ... before django.contrib
]
# ...
TEMPLATES = [
    {
        # ...
        'DIRS': ['templates'],
        # ...
    },
]
```

- Check database and timezone in `django_bpaml_strava_site/settings.py` (can use defaults = sqlite3 and UTC)

In `django_bpaml_strava_site/urls.py` link to new urls

```python
from django.contrib import admin
from django.urls import include, path
urlpatterns = [
    path('admin/', admin.site.urls),
    path('bpaml-strava/', include('django_bpaml_strava.urls')),
]
```

Create URLconf in `django_bpaml_strava/urls.py`

```python
from django.urls import path
from django_bpaml_strava.views import index_page

urlpatterns = [
  path('', index_page, name='index'),
]
```

Create the first view in `django_bpaml_strava/views.py`

```python
from django.http import HttpResponse
def index_page(request):
    return HttpResponse("<h1>BPAML Strava</h1>")
```

Can create a run configuration to easily run program

- name: runserver
- venv: should default to your local virtual environment
- module: manage
- script parameters: runserver
- (optional. slower but prevents stdout and stderr losing data) environment variables: PYTHONUNBUFFERED=1

Now can test at <http://localhost:8000/bpaml-strava/>

### Database setup <https://docs.djangoproject.com/en/5.2/intro/tutorial02/> `git checkout step03`

Create a data structure for Activity and Athlete.

- Activities need an athlete, date, start_time, title, location, description, duration, distance, polyline. Because each record could be in a different timezone need to store timezone information as well. In the future we will be capturing parkrun information as well.
- Athletes can extend the admin User table. Only extra field required is their parkrun id

Create database models in `django_bpaml_strava/models.py`

```python
from django.db import models
from django.contrib.auth.models import AbstractUser

class User(AbstractUser):
    """
    Extends Django User class and provides extra field for parkrun_id
    Strava ID also required but will be handled by allauth[socialaccount] 
    """
    # add extra fields to AbstractUser to make User
    # null=True means field can be empty in database
    # blank=True means field can be empty in django forms
    parkrun_id = models.IntegerField(null=True, blank=True)

    def __str__(self):
        return f"{self.email}<{self.first_name} {self.last_name}>"

# Need to define Activity second because it refers to User
class Activity(models.Model):
    athlete = models.ForeignKey(User, on_delete=models.CASCADE)
    activity_id = models.IntegerField(null=True)
    date = models.DateField("date of activity")
    start_time = models.DateTimeField("start date and time in UTC")
    start_time_local = models.DateTimeField("start date and time in local timezone", null=True, blank=True)
    timezone = models.CharField(max_length=200, default="UTC")
    title = models.CharField(max_length=200)
    location = models.CharField(max_length=200)  # parkrun location
    description = models.CharField(max_length=4000)
    parkrun_duration = models.DurationField(default=None, null=True, blank=True)
    strava_duration = models.DurationField(default=None, null=True, blank=True)
    distance = models.FloatField(default=0)  # metres
    polyline = models.CharField(max_length=4000)

    def __str__(self):
        return self.title
```

Make django_bpaml_strava show up in admin app by editing `django_bpaml_strava/admin.py`

```python
from django.contrib import admin

from django_bpaml_strava.models import Activity, User

admin.site.register(Activity)
admin.site.register(User)
```

Tell django to use our new User model for admin functions by adding the following to the bottom of `django_bpaml_strava_site/settings.py`

```python
# Override the provided user model for our use adding
# fields: strava_id, parkrun_id
# relationships: to many activities
AUTH_USER_MODEL = 'django_bpaml_strava.User'
```

- Make the database migrations for bpaml_strava with `python manage.py makemigrations`
- Create database tables for django_bpaml_strava and admin `python manage.py migrate` (git users do this also)

- From a terminal window create a superuser `python manage.py createsuperuser` (git users do this also)
- Check out admin section of your django site <http://localhost:8000/admin/>
- Check out bpaml strava section of site <http://localhost:8000/bpaml-strava/>

You can add some athletes in the admin interface. For now, we will just have the superuser.

#### Create a base template so our style can be consistent throughout `git checkout step04`

Base it on the admin base template. Create file `django_bpaml_strava/templates/django_bpaml_strava/base.html`

```html
{% extends 'admin/base_site.html' %}
{% load static %}
{% block extrastyle %}
<link rel="stylesheet" href="{% static 'django_bpaml_strava/bpaml.css' %}">
{% endblock %}
{% block title %}BPAML{% endblock %}
{% block branding %}
<div id="site-name"><a href="{% url 'index' %}">{{ site_header|default:_('BPAML Strava') }}</a></div>
{% include "admin/color_theme_toggle.html" %}
{% endblock %}
{% block content %}
<div class="bpaml-content">
    {% block aside %}
    <aside id="user-sidebar"></aside>
    {% endblock %}
    <div class="bpaml-main">
    {% block main %}{% endblock %}
    </div>
</div>
{% endblock %}
```

Create a style sheet `django_bpaml_strava/static/django_bpaml_strava/bpaml.css`. Notice the `static` folder for stylesheets and images.

```css
.bpaml-content {font-size: 1.0em;}
.bpaml-content .btnlink {padding: 7px; background: var(--button-bg); border: none; border-radius: 4px; color: var(--button-fg); vertical-align: middle; font-family: var(--font-family-primary); font-size: 0.8125rem;}
#user-sidebar .label {font-size: 0.7em;}
#user-sidebar {z-index: 15; flex: 0 0 275px; border-top: 1px solid transparent; border-right: 1px solid var(--hairline-color); background-color: var(--body-bg); overflow: auto;}
#user-sidebar table {width: 100%;}
#user-sidebar .module td {white-space: nowrap;}
#content > .bpaml-content {display: flex; flex: 1 0 auto;}
.bpaml-content .bpaml-main {padding-left: 10px;}
```

#### Create a template for <http://127.0.0.1:8000> with a couple of links in it

In our home page template `django_bpaml_strava/templates/bpaml_home.html` provide links to our app index page and the admin app index page. Notice that it is in the `templates` directory, not a subdirectory.

```html
{% extends 'django_bpaml_strava/base.html' %}
{% block main %}
<h1>BPAML</h1>
<ul>
    <li><a href="{% url 'index' %}">BPAML Strava</a></li>
    <li><a href="{% url 'admin:index' %}">BPAML Admin</a></li>
</ul>
{% endblock %}
```

In `django_bpaml_strava_site/urls.py`

```python
from django.urls import path
from django.views.generic import TemplateView  # new
urlpatterns = [
     # ...
     path('', TemplateView.as_view(template_name="bpaml_home.html")),  # new
]
```

In each new template remove the surrounding `<body>` tags and everything outside them and enclose remainder with

```html
{% extends 'django_bpaml_strava/base.html' %}
{% block main %}

{% endblock %}
```

Try it out.

### Display real data in our web app `git checkout step05`

Edit `django_bpaml_strava/views.py` to fetch the data we want

```python
from allauth.socialaccount.models import SocialAccount
from django.shortcuts import render, get_object_or_404

def index_page(request):
    """Find all athletes """
    list_social_accounts = SocialAccount.objects.filter(provider='strava').select_related('user')
    context = {'athletes': list_social_accounts}
    return render(request, 'django_bpaml_strava/athletes.html', context)

def athlete_page(request, strava_id):
    """Find just the one athlete with the supplied strava id"""
    a = get_object_or_404(SocialAccount, provider='strava', uid=strava_id)
    context = {'athlete': a}
    return render(request, 'django_bpaml_strava/athlete.html', context)
```


In `django_bpaml_strava/templates/django_bpaml_strava/athletes.html`

```html
{% extends 'django_bpaml_strava/base.html' %}
{% block main %}
<h1>BPAML Strava athletes</h1>
<table>
    <tr>
        <th>Name</th>
        <th>Strava ID</th>
        <th>Parkrun ID</th>
        <th>Activities</th>
        <th>action</th>
    </tr>
    {% for athlete in athletes %}
        <tr>
            <td>{{athlete.user.first_name}} {{athlete.user.last_name}}</td>
            <td>{{athlete.uid}}</td>
            <td>{{athlete.user.parkrun_id}}</td>
            <td>{{athlete.user.activity_set.count}}</td>
            <td>
                <a href="{% url 'view-activities' athlete.uid %}">view activities</a>
            </td>
        </tr>
    {% endfor %}
</table>
{% endblock %}
```

In `django_bpaml_strava/urls.py` create some functions which return these views.

```python
from django.urls import path
from django_bpaml_strava.views import index_page, athlete_page
urlpatterns = [
  path('', index_page, name='index'),
  path('athlete/<str:strava_id>/', athlete_page, name='athlete'),
]
```

Create `django_bpaml_strava/templates/django_bpaml_strava/athlete.html` the following two blocks

```html
{% extends 'django_bpaml_strava/base.html' %}
{% block main %}
{% if user.is_authenticated %}
<h2>Athlete</h2>
<table>
    <tr>
        <td>Name:</td>
        <td>{{ athlete.user.first_name }} {{ athlete.user.last_name }}</td>
    </tr>
    <tr>
        <td>Strava ID: </td>
        <td>{{ athlete.uid }}</td>
    </tr>
    <tr>
        <td>Parkrun ID:</td>
        <td>{{ athlete.user.parkrun_id }}</td>
    </tr>
</table>
<h2>Saved activities</h2>
<table>
    <tr>
        <th>ID</th>
        <th>Date</th>
        <th>Start time</th>
        <th>Description</th>
        <th>Distance</th>
        <th>Strava duration</th>
        <th>Parkrun duration</th>
        <th>Parkrun location</th>
        <th>Action</th>
    </tr>
    {% for activity in athlete.user.activity_set.all %}
    <tr>
        <td>{{ activity.activity_id }}</td>
        <td>{{ activity.date|date:"D d M Y" }}</td>
        <td>{{ activity.start_time_local|date:"D d M Y H:i" }} {{activity.timezone}}</td>
        <td>{{ activity.title }}</td>
        <td>{{ activity.distance }}</td>
        <td>{{ activity.strava_duration }}</td>
        <td>{{ activity.parkrun_duration }}</td>
        <td>{{ activity.parkrun_location }}</td>
        <td><a href="{% url 'delete-activity' athlete.uid activity.activity_id %}">delete</a></td>
    </tr>
    {% endfor %}
</table>
{% else %}
<p>Please sign in to update member details</p>
{% endif %}
{% endblock %}
{% block nav-breadcrumbs %}
<nav aria-label="Breadcrumbs">
  <div class="breadcrumbs">
    <a href="{% url 'index' %}">Athletes</a>
    &gt;
    <a href="{% url 'athlete' athlete.uid %}">{{ athlete.user.first_name }} {{ athlete.user.last_name }}</a>
  </div>
</nav>
{% endblock %}
```

This code demonstrates how to:

- show values from data model
- construct dynamic URLs
- override different blocks in templates being extended

Edit `django_bpaml_strava_site/settings.py` to add allauth components

```python
AUTHENTICATION_BACKENDS = [
    # Needed to log in by username in Django admin, regardless of `allauth`
    'django.contrib.auth.backends.ModelBackend',
    # `allauth` specific authentication methods, such as login by email
    'allauth.account.auth_backends.AuthenticationBackend',
]
INSTALLED_APPS = [
    'django_bpaml_strava.apps.DjangoBpamlStravaConfig',
    'allauth',
    'allauth.account',
    'allauth.socialaccount',
    'allauth.socialaccount.providers.strava',
    # ... before django.contrib
]
MIDDLEWARE = [
    # ...
    'allauth.account.middleware.AccountMiddleware',
]

# ... 
SOCIALACCOUNT_PROVIDERS = {'strava': {'SCOPE': ['read,activity:read']}}

LOGIN_REDIRECT_URL = "/bpaml-strava/"
LOGOUT_REDIRECT_URL = "/bpaml-strava/"

# Extend Django default logging to also log to console at level INFO rather than WARNING
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "INFO",
    },
}

DATE_FORMAT = 'D d M Y'
```

Now that we have changed `settings.py` to use allauth we need to migrate the data structure in the database 

```bash
python manage.py migrate
```

### Challenge - add the breadcrumbs to index page `git checkout step06`

Here are the breadcrumbs for the top navigation bar in `athletes.html`

```html
{% block nav-breadcrumbs %}
<nav aria-label="Breadcrumbs">
  <div class="breadcrumbs">
    <a href="{% url 'index' %}">Athletes</a>
  </div>
</nav>
{% endblock %}
```

## OAuth `git checkout step07`

## Create social application 

(git users do this also)
Create a strava api key for the manager of this app (see meetup 220)

Needed credentials are client ID and client secret <https://docs.allauth.org/en/latest/socialaccount/providers/strava.html>

In the project directory (same level as `django_bpaml_strava` and `django_bpaml_strava_site`) put the file `strava-key.json` that you created in meetup 220

Create a command line management command `django_bpaml_strava/management/commands/create_social_apps.py`

```python
import json
import pathlib
import sys
from django.core.management.base import BaseCommand
from allauth.socialaccount.models import SocialApp

STRAVA_KEY_PATH = pathlib.Path("strava-key.json")

def fetch_strava_key() -> dict[str, str | int]:
    """Fetches api key from file"""
    try:
        with STRAVA_KEY_PATH.open("r") as json_file:
            dct_api_key = json.load(json_file)
        return dct_api_key
    except FileNotFoundError:
        sys.exit(f'{STRAVA_KEY_PATH.resolve()} not found')


class Command(BaseCommand):
    help = 'Create social applications from key file or environment variables'

    def handle(self, *args, **options):
        strava_api_key = fetch_strava_key()
        social_apps = [
            {
                'provider': 'strava',
                'name': 'strava',
                'client_id': strava_api_key.get('client_id'),
                'secret': strava_api_key.get('client_secret'),
                'key': '',
                'settings': {},
            }
        ]

        for app_data in social_apps:
            if not app_data['client_id'] or not app_data['secret']:
                self.stdout.write(
                    self.style.ERROR(
                        f'Missing keys for {app_data["provider"]} in {STRAVA_KEY_PATH.resolve()}. {strava_api_key.get("client_id")=} {strava_api_key.get("client_secret")=}'
                    )
                )
                continue

            social_app, created = SocialApp.objects.get_or_create(
                provider=app_data['provider'],
                defaults={
                    'name': app_data['name'],
                    'client_id': app_data['client_id'],
                    'secret': app_data['secret'],
                    'key': app_data.get('key', ''),
                    'settings': app_data.get('settings', {}),
                }
            )

            if created:
                self.stdout.write(self.style.SUCCESS(f'Created social app: {app_data["name"]}'))
            else:
                social_app.client_id = app_data['client_id']
                social_app.secret = app_data['secret']
                social_app.save()
                self.stdout.write(self.style.WARNING(f'Updated social app: {app_data["name"]}'))
```

Create two empty files `django_bpaml_strava/management/__init__.py` and `django_bpaml_strava/management/commands/__init__.py`. This will tell python these directories are python packages.

Now run the management command to create the social application 

```bash
python manage.py create_social_apps
```

## configure links from app to allauth social accounts `git checkout step08`

Add the following to project `django_bpaml_strava_site/urls.py`

```python
from django.urls import include, path
urlpatterns = [
    # ...
    path('accounts/', include('allauth.urls')),
]
```

In `templates/django_bpaml_strava/base.html` load socialaccount at the top (after extends) `{% load socialaccount %}` and
add the following to the aside block

```html
{% block aside %}
<aside class="sticky" id="user-sidebar">
    <div class="module">
        <table>
            <caption>Authenticated user</caption>
            {% if user.is_authenticated %}
            <tr><td><div class="label">Email</div><div>{{user.email}}</div></td></tr>
            <tr><td><div class="label">Name</div><div>{{user.first_name}} {{user.last_name}}</div></td></tr>
            <tr><td><div><a class="btnlink" href="{% url 'index' %}">Update member details</a></div></td></tr>
            <tr><td><div><a class="btnlink" href="{% url 'account_logout' %}">Sign out</a></div></td></tr>
            {% else %}
            <tr><td><div>Not logged in.</div></td></tr>
            <tr><td><div>
                <a class="btnlink" href="{% provider_login_url 'strava' %}">Sign in with Strava</a>
                <a class="btnlink" href="{% url 'admin:login' %}?next={{request.path}}">Sign in with Django</a>
            </div></td></tr>
            {% endif %}
        </table>
    </div>
</aside>
{% endblock %}
```

Replace with the following in app urls `django_bpaml_strava/urls.py`

```python
from django.urls import path
from django_bpaml_strava.views import index_page, athlete_page, view_activities
urlpatterns = [
  path('', index_page, name='index'),
  path('athlete/<str:strava_id>/', athlete_page, name='athlete'),
  path('view-activities/athlete/<int:strava_id>', view_activities, name='view-activities'),
]
```

Add the following to app views `django_bpaml_strava/views.py` (imports at the top, defs at the bottom)

```python
from django.db.models import Prefetch
from django.contrib.auth.decorators import login_required
from django_bpaml_strava.models import Activity, User

# ...

def social_account_with_sorted_activities(strava_id):
    social_account = (SocialAccount.objects.filter(uid=strava_id, provider='strava')
                      .select_related("user")
                      .prefetch_related(
                          Prefetch(
                            "user__activity_set",
                            queryset=Activity.objects.order_by("start_time")
                          )
                        )
                      .first())
    return social_account

@login_required
def view_activities(request, strava_id):
    # Get the stored, and possibly refreshed, access token for this athlete
    social_account = social_account_with_sorted_activities(strava_id)
    context = {'athlete': social_account}
    return render(request, 'django_bpaml_strava/athlete.html', context)
```

Now try logging in using strava or django and see how it creates a new account
