# geology-neuro-module

## How to install

1. Install `python3.8` from [here](https://www.python.org/downloads/release/python-388/)
   or `brew install python@3.8`
2. Go to project folder: `cd /your/path/to/geology-neuro-module`
3. Setup virtual environment: `python3.8 -m venv venv`
4. Activate virtual environment: `source venv/bin/activate`
5. Install required packages: `python -m pip install -r requirements.txt`
6. Copy [docs/golden_CONFIG.cfg](docs/golden_CONFIG.cfg) in root and rename to `CONFIG.cfg`
7. Set up values in `CONFIG.cfg`
8. Deactivate virtual environment: `deactivate`

## How to configure DATABASE

### SQLite

* SQLite comes pre-installed with `python`

1. In `CONFIG.cfg`, `[DATABASE]` section set up:
   1. `DB_DRIVER = sqlite`
   2. `DB_HOST = /your/path/to/database.db`
   3. Empty other values
2. Init (or upgrade existing) database structure: `alembic upgrade head`

### PostgreSQL

1. Install PostreSQL from [here](https://www.postgresql.org/download/)
## How to launch

1. Go to project folder `cd /your/path/to/geologyAPI`
2. Make sure you are in virtual environment: `source venv/bin/activate`
3. Start HTTP-server: `gunicorn neuroAPI:server`
