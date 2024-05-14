#!/bin/bash

# Activate the virtual environment if necessary
# source antenv/bin/activate

# Run the Flask app with Gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app
