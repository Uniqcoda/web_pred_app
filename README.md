# Skin Lesion Classification

This is a Python Flask app for skin cancer prediction.

### Set up

Create virtual environment.

```
$ python3 -m venv .venv
```

```
$ . .venv/bin/activate
```

Install dependencies.

#### first time?

```
$ pip install flask opencv-python numpy tensorflow flask-restful flask-cors gunicorn
```

Short cut

```
$ python3 -m venv .venv; . .venv/bin/activate; pip install flask opencv-python numpy tensorflow flask-restful flask-cors gunicorn
```
Create requirements file
```
pip freeze > requirements.txt
```

#### subsequent time?

```
$ pip install -r requirements.txt
```

### Dev Run

Export flask directory.

```
$ export FLASK_APP=app.py
```

Run app

```
$ flask run
```

### Production Run

```
$ gunicorn --workers=4 --bind 0.0.0.0 app:app
```

_Some sample images are available in `/sample_images` to upload and test the app._
