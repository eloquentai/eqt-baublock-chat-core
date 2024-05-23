# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy only the relevant files
COPY app.py .
COPY pyproject.toml .
COPY public/ public/
COPY .chainlit/ .chainlit/
COPY chainlit.md .

# Install Poetry for dependency management
RUN pip install poetry

# Disable virtual env creation by poetry and install dependencies from pyproject.toml
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Expose port 8000 for the application
EXPOSE 8000 

# Command to run the application
CMD ["chainlit", "run", "app.py", "-w"]
