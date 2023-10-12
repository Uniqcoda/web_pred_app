# Skin Lesion Classification

This is a Python Flask app for skin cancer prediction.

### Set up

Create virtual environment.

```
$ python3 -m venv .venv
```

Activate virtual environment.

```
$ . .venv/bin/activate
```

Install pipenv package manager

```
pip install pipenv
```

Install dependencies.

```
$ pipenv install flask opencv-python numpy tensorflow flask-restful flask-cors gunicorn
```

Export flask directory.

```
$ export FLASK_APP=app.py
```

### Dev Run

```
$ flask run
```

### Production Run

```
$ gunicorn --workers=4 --bind 0.0.0.0:5000 app:app
```

_Some sample images are available in `/sample_images` to upload and test the app._
