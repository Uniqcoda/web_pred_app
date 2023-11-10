# Skin Lesion Classification

This is a Python Flask app for skin cancer prediction.

## Set up

#### Create virtual environment.

```
python3 -m venv .venv
```

```
source .venv/bin/activate
```

Install dependencies.

#### first time?

```
pip install flask opencv-python numpy tensorflow gunicorn
```

Short cut

```
python3 -m venv .venv; source .venv/bin/activate; pip install flask opencv-python numpy tensorflow gunicorn
```
Create requirements file
```
pip freeze > requirements.txt
```

#### subsequent time?

```
pip install -r requirements.txt
```

## Dev Run
```
flask run
```

## Production Run

```
gunicorn --workers=4 --bind 0.0.0.0 app:app
```

_Some sample images are available in `/sample_images` to upload and test the app._

## Run with Docker

```
docker compose up --build
```
