# CartPole-NEAT
In this project I aim to implement [NEAT](https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies) in python. Then I will try solve the CartPole problem using it. I will use [OpenAI Gym's environment](https://gym.openai.com/envs/CartPole-v0/) for this. I also plan to make this program multi-threaded so that I can have multiple runs (with possibly different configurations) running in parallel.

I also want to make a supporting web application that will handle real-time plotting as I have found [matplotlib](https://matplotlib.org/) to be a bit lacking in that area. This as an opportunity to practice some web development skills and learn some new tech. This web application would be a RESTful application served locally from a [flask](http://flask.pocoo.org/) server running in [docker](https://www.docker.com/) container, backed by [MongoDB](https://www.mongodb.com/). For the front end of things I will probably use a combination of Twitter [Bootstrap](https://getbootstrap.com/), [React](https://reactjs.org/), and [JavaScript Live](https://canvasjs.com/html5-javascript-dynamic-chart/).

Check out the [projects page](https://github.com/eight0153/CartPole-NEAT/projects) to see the progress of brainstorming, planning, and implementation. This project is split up into three subprojects: the actual genetic algorithm implementation, the RESTful API that provides access to test data, and the dashboard that provides visualizations of previous runs and a real time visualization test runs that are in progress. 
