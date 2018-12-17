# Flask API
The flask API will be a RESTful API serving training data generated by the NEAT genetic algorithm. It is run as a containerised application with Docker.

## Getting Started
1. Make sure you have [Docker](https://docs.docker.com/install/) and [Docker Compose](https://docs.docker.com/compose/install/) installed.
2. Open your terminal at 'api/' or change it to 'api/' from the repository root directory:
    ```shell
    $ cd api/
    ```
3. Setup the .env file. See the following [section](#configuring-environment-settings) on configuring environment settings. 
3. Build with Docker Compose:
    ```shell
    $ docker-compose build
    ```
4. Start the API:
    ```shell
    $ docker-compose up -d
    ```
5. Test the API:
    ```shell
    $ curl http://127.0.0.1:5000/
    {
        "hello": "world"
    }
    ```
    or go to http://127.0.0.1:5000/ in your web browser.
6. Open psql on the PostgreSQL server:
    ```shell
    $ docker exec -it api_postgres_1 sh
    / # psql ${POSTGRES_DB} --username=${POSTGRES_USER}
    ```
    You can quit psql with ```\q``` and exit the docker container with ```exit```.
7.Stop the API:
    ```shell
    $ docker-compose stop
    ```
## Configuring Environment Settings
The flask server and PostgreSQL database require some environment variables to be set. These can be passed to Docker Compose via a .env file. To ensure that everything is setup correctly follow these steps:
1. Create the file ```.env``` in the ```api/``` directory.
    ```shell
    $ cd api/
    $ touch .env
    ```
2. Fill in the following fields:
    1. POSTGRES_USER
    2. POSTGRES_PASSWORD
    3. POSTGRES_DB
    4. POSTGRES_PORT
    5. FLASK_PORT

    Your final ```.env``` file may look like this:
    ```
    POSTGRES_USER=AzureDiamond
    POSTGRES_PASSWORD=hunter2
    POSTGRES_DB=bash_irc_chat_log
    POSTGRES_PORT=5432
    FLASK_PORT=5000
    ```