from flask_marshmallow import Marshmallow
from flask_sqlalchemy import SQLAlchemy
from marshmallow import fields

ma = Marshmallow()
db = SQLAlchemy()


# noinspection PyTypeChecker
class Run(db.Model):
    id = db.Column(db.String(16), primary_key=True)
    start_date = db.Column(db.TIMESTAMP, server_default=db.func.current_timestamp(), nullable=False)
    end_date = db.Column(db.TIMESTAMP)

    def __init__(self, id):
        self.id = id


class RunSchema(ma.Schema):
    id = fields.String()
    start_date = fields.DateTime()
    end_date = fields.DateTime()
