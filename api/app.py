from flask import Blueprint
from flask_restful import Api

from resources.hello import Hello
from resources.runs import Run, Runs, RunFinished

api_bp = Blueprint('api', __name__)
api = Api(api_bp)

api.add_resource(Hello, '/hello', '/hello/<name>')
api.add_resource(Runs, '/runs')
api.add_resource(Run, '/runs/<id>')
api.add_resource(RunFinished, '/runs/<id>/finished')
