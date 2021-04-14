[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg)](code_of_conduct.md)

# geologyAPI

## neuroAPI

### How to install

1. Install `python3.8` from [here](https://www.python.org/downloads/release/python-388/) or `brew install python@3.8`
2. Go to project folder: `cd /your/path/to/geologyAPI`
3. Setup virtual environment: `python3.8 -m venv venv`
4. Activate virtual environment: `source venv/bin/activate`
5. Install required packages: `python -m pip install -r requirements.txt`
6. Reconfigure values in [pyCONFIG.cfg](pyCONFIG.cfg)
7. Deactivate virtual environment: `deactivate`

### How to launch

1. Go to project folder `cd /your/path/to/geologyAPI`
2. Make sure you are in virtual environment: `source venv/bin/activate`
3. Start HTTP-server: `gunicorn neuroAPI:server`
