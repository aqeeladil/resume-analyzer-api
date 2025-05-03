# Resume Analyzer API

For the video demo, [click here](https://www.awesomescreenshot.com/video/38436796?key=f0b540f8416dc4089e47a568b2f90c15).

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
