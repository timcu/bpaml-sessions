# MeetUp 226 - Beginners Python and Machine Learning - Wed 19 Nov 2025 - Django and Strava tokens with PyCharm

## Part 1 - 15 Oct 2025 - Writing a Django Web App and authenticating with Strava - Use Strava auth to authenticate in a Django web app

See <https://github.com/timcu/bpaml-sessions/blob/master/online/meetup224_tim_django_strava_auth.md>

## Part 2 - 19 Nov 2025 - Saving, refreshing and using Strava tokens

Links:

- Youtube: <https://youtu.be/>
- Github:  <https://github.com/timcu/bpaml-sessions/blob/master/online/meetup226_tim_django_strava_token.md>
- Meetup:  <https://www.meetup.com/beginners-python-machine-learning/events/>
- Source:  <https://github.com/timcu/django_bpaml_strava_site> 
## References

- <https://www.jetbrains.com/pycharm/>
- <https://docs.djangoproject.com/en/5.2/intro/tutorial01/>

## Procedure

### Create project and app <https://docs.djangoproject.com/en/5.2/intro/tutorial01/>

To start where we left off in part 1, check out the source code from GitHub into a PyCharm project

- `git clone https://github.com/timcu/django_bpaml_strava_site`
- `git checkout step08`


- Open as a PyCharm project `django_bpaml_strava_site`
  - Make sure you use a new virtual environment either venv, conda depending on your python installation
- Open Terminal window
- `pip install -r requirements.txt` This will also install Django, allauth with Strava Auth and other dependencies
- Create a database to store data `python manage.py migrate`
- Create a superuser `python manage.py createsuperuser`
- Add your personal strava client secret `strava-key.json`
- Create the social apps `python manage.py create_social_apps`

Create a run configuration for `runserver`

- name: runserver
- venv: should default to your local virtual environment
- module: manage
- script parameters: runserver
- (optional. slower but prevents stdout and stderr losing data) environment variables: PYTHONUNBUFFERED=1

Try it out at <http://localhost:8000> and try logging in using your strava account



## Use strava tokens to get activity data (`git checkout step09` for lazy typists)

We need to do all of the next few steps before we can test and see how it is going

1. Tell Django allauth to save Strava tokens so they can be reused
2. Create a function to refresh Strava tokens if required. Django allauth does not do this automatically
3. Create function in `django_bpaml_strava/views.py` to fetch activities from Strava
4. Create url in `django_bpaml_strava/urls.py` to trigger fetching
5. Create hyperlink in `django_bpaml_strava/templates/django_bpaml_strava/athlete.html` to that url and html table to display the fetched activities

### Save Strava tokens

First we need to tell Django allauth to save refreshed strava tokens. Otherwise only the user will need to log in every time they want to use the token. In `django_bpaml_strava_site/settings.py` add the following line

```python
SOCIALACCOUNT_STORE_TOKENS = True
```

### Refresh Strava tokens

Secondly, create a module `django_bpaml_strava/strava_token.py` with a method to fetch the token and refresh it if required. Similar to the one developed in meetup 220.

```python
# Standard libraries
import datetime
import logging
# Third-party libraries
from allauth.socialaccount.models import SocialToken, SocialAccount, SocialApp
import requests

logger = logging.getLogger(__name__)

def fetch_strava_token(strava_id):
    """Retrieves token from database and, if expired, refreshes using Strava api"""
    social_account = SocialAccount.objects.get(uid=strava_id, provider='strava')
    logger.info(f"Found athlete {social_account.uid}")
    # Check if user has any social tokens
    social_tokens = SocialToken.objects.filter(account=social_account)
    logger.info(f"Found {len(social_tokens)} social tokens")
    # Get the stored access token for this athlete. social_tokens should actually only be one token
    strava_token = social_tokens.first()
    if strava_token.expires_at <= datetime.datetime.now(datetime.timezone.utc):
        logger.info(f"token expired at {strava_token.expires_at} - refreshing")
        # https://developers.strava.com/docs/authentication/ "Refreshing Expired Access Tokens"
        token_url = 'https://www.strava.com/api/v3/oauth/token'
        strava_app = SocialApp.objects.get(provider='strava')
        payload = {
            'client_id': strava_app.client_id,
            'client_secret': strava_app.secret,
            'grant_type': 'refresh_token',
            'refresh_token': strava_token.token_secret
        }
        response = requests.post(token_url, data=payload)
        if response.ok:
            new_tokens = response.json()
            # Update the api_key dictionary with the new access and refresh tokens and expiry time
            strava_token.token = new_tokens['access_token']
            strava_token.token_secret = new_tokens['refresh_token']
            strava_token.expires_at = datetime.datetime.fromtimestamp(new_tokens['expires_at'], datetime.timezone.utc)
            logger.info("Token refreshed successfully.")
            # Save the refreshed key for a later session
            strava_token.save()
        else:
            logger.info(f"Error refreshing token: {response.status_code} - {response.text}")
    return strava_token
```

### Create functions in `django_bpaml_strava/views.py` to fetch activities from Strava

Typical activities data (extract only)

```python
activity_extract = {'achievement_count': 0,
  'athlete': {'id': 43060840, 'resource_state': 1},
  'distance': 5026.5,
  'elapsed_time': 3187,
  'end_latlng': [-27.5, 153.01],
  'id': 14014834943,
  'kudos_count': 12,
  'map': {'id': 'a14014834943', 'resource_state': 2, 'summary_polyline': 'jfy...CLAC'},
  'max_speed': 15.143,
  'moving_time': 2720,
  'name': 'Cool down',
  'sport_type': 'Run',
  'start_date': '2025-03-28T21:28:36Z',
  'start_date_local': '2025-03-29T07:28:36Z',
  'start_latlng': [-27.5, 153.02],
  'timezone': '(GMT+10:00) Australia/Brisbane',
  'type': 'Run',
  'utc_offset': 36000.0,
  'visibility': 'everyone',
}
```

Function to fetch activities from Strava using athlete's Strava token

```python
import datetime
import logging
import requests
import zoneinfo
from django_bpaml_strava.strava_token import fetch_strava_token

logger = logging.getLogger(__name__)
BASE_TZ = zoneinfo.ZoneInfo("Australia/Brisbane")

# ...
def fetch_activities_from_strava(strava_id):
    """Fetch all the strava activities for a user in first three months of current year without saving them"""
    social_token = fetch_strava_token(strava_id)

    year = datetime.datetime.now(BASE_TZ).year
    after = int(datetime.datetime(year, 1, 1, 0, 0, tzinfo=BASE_TZ).timestamp())
    before = int(datetime.datetime(year, 4, 1, 0, 0, tzinfo=BASE_TZ).timestamp())

    # Define the endpoint and headers
    url = 'https://www.strava.com/api/v3/athlete/activities'
    headers = {'Authorization': f'Bearer {social_token.token}'}

    # Define parameters for the request
    params = {
        'before': before,
        'after': after,
        'page': 1,
        'per_page': 200,
    }

    # Make the GET request with parameters
    response = requests.get(url, headers=headers, params=params)

    # Check if the request was successful
    if response.ok:
        activity_data = response.json()
        logger.info(f"Athlete activities for authorized user: {len(activity_data)}")
        return activity_data
    else:
        logger.warning(f"Error requesting activities from strava {response.status_code}: {response.text}")
        return None

@login_required
def fetch_and_view_activities(request, strava_id):
    """Fetch activities from strava, no filtering, and display asking user which ones should be saved
    
    Triggered by URL 'bpaml-strava/show-unsaved-activities-available-on-strava'
    """
    list_strava_activities = fetch_activities_from_strava(strava_id)
    if list_strava_activities is None:
        return index_page(request)
    social_account = social_account_with_sorted_activities(strava_id)
    # omit the ones already saved
    set_activity_id = set(a.activity_id for a in social_account.user.activity_set.all())
    logger.info(f"{set_activity_id=}")
    for i in range(len(list_strava_activities)-1, 0, -1):
        if list_strava_activities[i]['id'] in set_activity_id:
            del list_strava_activities[i]
    list_new_strava_activities = [d for d in list_strava_activities if d['id'] not in set_activity_id]
    # display the results
    context = {'athlete': social_account, 'list_strava_activities': list_new_strava_activities}
    return render(request, 'django_bpaml_strava/athlete.html', context)

```

### Create url in `django_bpaml_strava/urls.py`

```python
# ...
from django_bpaml_strava.views import fetch_and_view_activities

urlpatterns = [
    # ...
  path('view-unsaved-activities-available-on-strava/athlete/<int:strava_id>', fetch_and_view_activities, name='view-unsaved-activities-available-on-strava'),
]
```

### Create hyperlink and table in `django_bpaml_strava/templates/django_bpaml_strava/athlete.html`

```html
<!-- after existing table showing saved activities -->
{% if list_strava_activities %}
<h2>Other strava activities</h2>
<table>
    <tr>
        <th>ID</th>
        <th>Date</th>
        <th>Start time</th>
        <th>Description</th>
        <th>Distance</th>
        <th>Strava duration</th>
        <th>Action</th>
    </tr>
    {% for activity in list_strava_activities %}
    <tr>
        <td>{{ activity.id }}</td>
        <td>{{ activity.start_date_local }}</td>
        <td>{{ activity.moving_time }}</td>
        <td>{{ activity.name }}</td>
        <td>{{ activity.distance }}</td>
        <td>{{ activity.elapsed_time }}</td>
        <td>TODO hyperlink to save</td>
    </tr>
    {% endfor %}
</table>
{% else %}
<p><a href="{% url 'view-unsaved-activities-available-on-strava' athlete.uid %}">view unsaved activities available on strava</a> </p>
{% endif %}
```

## Able to save one activity in database `git checkout step10`

Add a function to save an activity `django_bpaml_strava/views.py` and a function triggered by url `save-activity`

```python
def create_activity_from_strava(user: User, dct_activity):
    """
    Given the json data for a single activity from Strava (already converted to a dict)
    create an Activity record linked to the correct User and save in the database.
    """
    list_timezone = dct_activity['timezone'].split(' ')
    timezone = list_timezone[1]
    start_time = datetime.datetime.strptime(dct_activity["start_date"], "%Y-%m-%dT%H:%M:%SZ").astimezone(zoneinfo.ZoneInfo(timezone))
    start_time_local = datetime.datetime.strptime(dct_activity["start_date_local"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=zoneinfo.ZoneInfo(timezone))
    start_date = start_time.date()
    logger.info(f'{start_time:%d-%b-%Y %H:%M} {start_time_local:%d-%b-%Y %H:%M %z} {dct_activity["distance"] / 1000:6.1f}km {dct_activity["name"]}')
    activity = Activity.objects.create(
        athlete=user,
        activity_id=dct_activity["id"],
        date=start_date,
        start_time=start_time,
        start_time_local=start_time_local.replace(tzinfo=None),
        timezone=timezone,
        distance=dct_activity["distance"],
        title=dct_activity["name"],
        strava_duration=datetime.timedelta(seconds=dct_activity["elapsed_time"]),
        polyline=dct_activity["map"]["summary_polyline"],
    )
    activity.save()

@login_required
def save_activity(request, strava_id, activity_id):
    social_account = social_account_with_sorted_activities(strava_id)
    social_token = fetch_strava_token(strava_id)
    # Define the endpoint and headers to fetch a single activity
    url = f'https://www.strava.com/api/v3/activities/{activity_id}'
    headers = {'Authorization': f'Bearer {social_token.token}'}

    # Define parameters for the request. We don't need all efforts for this activity
    params = {
        'include_all_efforts': False,
    }

    # Make the GET request with parameters for single activity
    response = requests.get(url, headers=headers, params=params)

    # Check if the request was successful
    if response.ok:
        dct_activity = response.json()
        logger.info(f"Activity for authorized user: {dct_activity}")
        # add activity to user
        create_activity_from_strava(social_account.user, dct_activity)
    else:
        logger.warning(f"Error requesting activities from strava {response.status_code}: {response.text}")
    # requery db to include new activity
    return view_activities(request, strava_id)
```

Set up the URL for this action `django_bpaml_strava/urls.py`

```python
from django_bpaml_strava.views import save_activity

urlpatterns = [
  # ...
  path('save-activity/athlete/<int:strava_id>/activity/<int:activity_id>', save_activity, name='save-activity'),
]
```

Put hyperlink to call this URL in template `django_bpaml_strava/templates/django_bpaml_strava/athlete.html`. Replace text "TODO hyperlink to save" with 

```html
<a href="{% url 'save-activity' athlete.uid activity.id %}">save</a>
```

## Able to delete one activity from database

Add a function to save an activity `django_bpaml_strava/views.py` and a function triggered by url `delete-activity`

```python
@login_required
def delete_activity(request, strava_id, activity_id):
    social_account = social_account_with_sorted_activities(strava_id)
    for a in social_account.user.activity_set.all():
        if a.activity_id == activity_id:
            a.delete()
            logger.info(f"Deleted activity {a.activity_id}")
            break
        logger.info(f"activity {a.activity_id} != {activity_id}")
    else:
        logger.error("Activity not found")
    context = {'athlete': social_account}
    return render(request, 'django_bpaml_strava/athlete.html', context)
```

Set up the URL for this action `django_bpaml_strava/urls.py`

```python
from django_bpaml_strava.views import delete_activity
#...
  path('delete-activity/athlete/<int:strava_id>/activity/<int:activity_id>', delete_activity, name='delete-activity'),
```

Now try it out <http://127.0.0.1:8000/bpaml-strava/>

## Prevent undesriable consequences of page refreshes

Notice how refreshing the page after saving or deleting repeats the last action. A common web app pattern is PRG, POST, Redirect, Get. We don't have the initial post but we can 
still redirect and get the page to ensure the address bar has a benign url that can be refreshed without danger.

```python
from django.shortcuts import redirect
# ... following line replaces last line of save_activity and last two lines of delete_activity in views.py 
    return redirect('view-activities', strava_id=strava_id)
```

### Able to save bulk activities in database `git checkout step11`

Edit `django_bpaml_strava/views.py`. Extra logic on fetching activities to just look at those which might be parkruns on a Saturday starting before 7:10am and bteween 4700m and 5300m

```python
@login_required
def fetch_and_save_activities(request, strava_id):
    """Fetch activities from strava, filter out non-parkrun events and save the rest if nothing else
    already saved for that date"""
    lst_strava_activities = fetch_activities_from_strava(strava_id)
    if lst_strava_activities is None:
        return index_page(request)
    social_account = social_account_with_sorted_activities(strava_id)
    set_saturday = set(a.date for a in social_account.user.activity_set.all())
    for dct_activity in lst_strava_activities:
        # timezone looks like '(GMT+10:00) Australia/Brisbane'
        list_timezone = dct_activity['timezone'].split(' ')
        timezone = list_timezone[1]
        start_time = datetime.datetime.strptime(dct_activity["start_date"], "%Y-%m-%dT%H:%M:%SZ").astimezone(
            zoneinfo.ZoneInfo(timezone))
        start_date = start_time.date()
        latest_start_time = start_time.replace(hour=7, minute=10, second=0, microsecond=0)
        # Find last start on each Saturday before 7:10am local time that is between 4.7km and 5.3km
        if start_date not in set_saturday and start_time.weekday() == 5 and start_time < latest_start_time and 4700 < dct_activity["distance"] < 5300:
            set_saturday.add(start_date)
            create_activity_from_strava(social_account.user, dct_activity)
    return redirect('view-activities', strava_id=strava_id)
```

Create url in `django_bpaml_strava/urls.py`

```python
from django_bpaml_strava.views import fetch_and_save_activities

urlpatterns = [
    # ...
  path('save-activities/athlete/<int:strava_id>', fetch_and_save_activities, name='save-activities'),
]
```

Create hyperlink in template `django_bpaml_strava/templates/django_bpaml_strava/athlete.html`

```html
<p>
    <a href="{% url 'save-activities' athlete.uid %}">fetch Saturday activities for this athlete</a>
</p>
```

### Able to delete all activities for an athlete from database `git checkout step12`

Edit `django_bpaml_strava/views.py`. 

```python
@login_required
def delete_activities(request, strava_id):
    social_account = social_account_with_sorted_activities(strava_id)
    social_account.user.activity_set.all().delete()
    return redirect('view-activities', strava_id=strava_id)
```

Create url in `django_bpaml_strava/urls.py`

```python
from django_bpaml_strava.views import delete_activities

urlpatterns = [
    # ...
  path('delete-activities/athlete/<int:strava_id>', delete_activities, name='delete-activities'),
]
```

Create hyperlink in template `django_bpaml_strava/templates/django_bpaml_strava/athlete.html`

```html
<p>
    ...| <a href="{% url 'delete-activities' athlete.uid %}">delete saved activities for this athlete</a>
</p>
```
