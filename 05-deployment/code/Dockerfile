# 1. Choose the base Python image
FROM python:3.14.7-slim

# 2. Install Pipenv
RUN pip install pipenv

# 3. Create/select the working directory
WORKDIR /app

# 4. Copy the dependency definitions into /app inside the image
COPY ["Pipfile", "Pipfile.lock", "./"]

# 5. Install the locked dependencies into the container's Python environment
# --system: install into the container's Python environment, not a virtualenv
# --deploy: stop if the lockfile is out of sync with the Pipfile
RUN pipenv install --system --deploy

# 6. Copy the Flask application and its trained model into /app
COPY ["predict.py", "model_C=1.0.bin", "./"]

# 7. Document that the app listens on port 9696 inside the container
EXPOSE 9696

# 8. Automatically start Gunicorn when the container starts
# --bind=0.0.0.0:9696: listen on all network interfaces, on port 9696
# predict:app: load the "app" object from predict.py
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"] 