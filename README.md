# geology-neuro-module

## How to install

1. Install `python3.9` from [here](https://www.python.org/downloads/release/python-390/)
   or `brew install python@3.9`
1. Install `python3-dev`, `libpq-dev`, `build-essential`
2. Go to project folder: `cd /your/path/to/geology-neuro-module`
3. Setup virtual environment: `python3.9 -m venv venv`
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
2. Go to project folder: `cd /your/path/to/geology-neuro-module`
3. Activate virtual environment: `source venv/bin/activate`
4. Init (or upgrade existing) database structure: `alembic upgrade head`
5. Deactivate virtual environment: `deactivate`

### PostgreSQL

1. Install PostreSQL from [here](https://www.postgresql.org/download/)
2. Create empty database with locale = `*.UTF-8`
3. Set up `[DATABASE]` section in `CONFIG.cfg`:
   1. `DB_DRIVER = postgresql`
   2. `DB_USER = postgres` (by default)
   1. `DB_PASS = <password>`
   2. `DB_HOST = <pg_host>:<pg_port>`
   3. `DB_NAME = <your_db_name>`
2. Go to project folder: `cd /your/path/to/geology-neuro-module`
3. Activate virtual environment: `source venv/bin/activate`
4. Init (or upgrade existing) database structure: `alembic upgrade head`
5. Deactivate virtual environment: `deactivate`

## How to launch

1. Go to project folder `cd /your/path/to/geologyAPI`
2. Make sure you are in virtual environment: `source venv/bin/activate`
3. Start HTTP-server: `gunicorn neuroAPI:server`
