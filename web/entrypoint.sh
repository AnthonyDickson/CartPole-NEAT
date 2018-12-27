#!/usr/bin/env bash

export PYTHONPATH=.
python -m api.migrate db init
python -m api.migrate db migrate
python -m api.migrate db upgrade
python -m api.run
