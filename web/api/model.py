from flask_marshmallow import Marshmallow
from flask_sqlalchemy import SQLAlchemy
from marshmallow import fields

ma = Marshmallow()
db = SQLAlchemy()


# noinspection PyTypeChecker
class Run(db.Model):
    """A training run of NEAT."""
    id = db.Column(db.CHAR(16), primary_key=True)
    start_date = db.Column(db.TIMESTAMP, server_default=db.func.current_timestamp(), nullable=False)
    end_date = db.Column(db.TIMESTAMP)

    def __init__(self, id):
        self.id = id


class RunSchema(ma.Schema):
    id = fields.String()
    start_date = fields.DateTime()
    end_date = fields.DateTime()


class Node(db.Model):
    """A node in a neural network computational graph."""
    object_id = db.Column(db.Integer, primary_key=True)
    run_id = db.Column(db.String(16), db.ForeignKey('run.id'), nullable=False, primary_key=True)
    id = db.Column(db.Integer, nullable=False)
    type = db.Column(db.String(8), nullable=False)
    activation = db.Column(db.String(8), nullable=False)


class NodeSchema(ma.Schema):
    object_id = fields.Integer()
    run_id = fields.String()
    id = fields.Integer()
    type = fields.String()
    activation = fields.String()


