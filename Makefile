# Define your virtual environment and flask app
VENV = venv
FLASK_APP = app.py
PYTHON = python3.9  # or python3.10 depending on what you have installed

# Install dependencies
install:
	$(PYTHON) -m venv $(VENV)
	./$(VENV)/bin/pip install --upgrade pip
	./$(VENV)/bin/pip install -r requirements.txt

# Run the Flask application
run:
	./$(VENV)/bin/flask run --port 3000

# Clean up virtual environment
clean:
	rm -rf $(VENV)

# Reinstall all dependencies
reinstall: clean install