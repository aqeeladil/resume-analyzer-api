# Resume Analyzer API

For the video demo, [click here](https://www.awesomescreenshot.com/video/38392630?key=0c1a192ab8326edbaa3d2b9fc60584c9).

```bash
# Prerequisites
pip install django djangorestframework djangorestframework-simplejwt pymupdf django-cors-headers django-environ gunicorn scikit-learn

# Project Setup
django-admin startproject resume_analyzer
cd resume_analyzer
python manage.py startapp api

python manage.py makemigrations
python manage.py migrate
python manage.py runserver
```
