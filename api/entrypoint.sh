#!/usr/bin/env bash

python migrate.py db init
python migrate.py db migrate
python migrate.py db upgrade
python run.py
