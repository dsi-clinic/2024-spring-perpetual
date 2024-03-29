#!/bin/bash

# Log script start
echo "Starting setup script."

# Configure script to exit when any command fails
set -e

# Monitor last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG

# Log error message upon script exit
trap '[ $? -eq 1 ] && echo "Backend initialization failed."' EXIT

# Parse command line arguments
migrate=false
use_uwsgi_server=false
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --migrate) migrate=true; shift ;;
        --use_uwsgi_server) use_uwsgi_server=true; shift ;;
        *) echo "Unknown command line parameter received: $1"; exit 1 ;;
    esac
done

# Wait for database server to accept connections
python3 wait_for_postgres.py

# Perform model migrations if indicated 
# (WARNING: Defaults to "yes" for all questions)
if $migrate ; then
    echo "Creating database migrations from Django models."
    yes | ./manage.py makemigrations

    echo "Applying migrations to database."
    yes | ./manage.py migrate
fi

# Run server
if $use_uwsgi_server ; then
    echo "Running UWSGI server."
    uwsgi --http ":8080" \
        --chdir "/usr/src/backend" \
        --module "config.wsgi:application" \
        --uid "1000" \
        --gid "2000" \
        --http-timeout "1000"
else
    echo "Running default development server."
    ./manage.py runserver 0.0.0.0:8080
fi