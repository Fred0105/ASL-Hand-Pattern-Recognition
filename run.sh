#!/bin/bash
# Run the application using the validated virtual environment
cd "$(dirname "$0")/frontend"
../.venv/bin/python app.py
