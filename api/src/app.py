import os

from flask import Flask, abort
from flask_restful import Resource, Api
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ['SQLALCHEMY_DATABASE_URI']
db = SQLAlchemy(app)
api = Api(app)

class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}

api.add_resource(HelloWorld, '/')


class UserAPI(Resource):
    def get(self, id=None):
        if id:
            user = User.query.filter_by(id=id).first()

            if user is None:
                abort(404)
            else:
                return {'data': user.to_json()}
        else:
            return {'data': [user.to_json() for user in User.query.all()]}

api.add_resource(UserAPI, '/users', '/users/<id>')

class User(db.Model):
    global db

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)

    def __repr__(self):
        return '<User %r>' % self.username

    def to_json(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email
        }

if __name__ == '__main__':
    db.create_all()

    app.run(debug=True, host='0.0.0.0')