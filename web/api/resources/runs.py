from flask import request
from flask_restful import Resource

from api.model import Run as RunModel
from api.model import db, RunSchema

runs_schema = RunSchema(many=True)
run_schema = RunSchema()


class Runs(Resource):
    def post(self):
        json_data = request.get_json(force=True)

        if not json_data:
            return {'message': 'No input data provided'}, 400
        # Validate and deserialize input
        data, errors = run_schema.load(json_data)

        if errors:
            return errors, 422

        run = RunModel.query.filter_by(id=data['id']).first()

        if run:
            return {'message': 'Run already exists'}, 400

        run = RunModel(
            id=json_data['id']
        )

        db.session.add(run)
        db.session.commit()

        result = run_schema.dump(run).data

        return {"status": 'success', 'data': result}, 201

    def get(self):
        runs = RunModel.query.all()
        runs = runs_schema.dump(runs).data

        return {'status': 'success', 'data': runs}, 200


class Run(Resource):
    def get(self, id):
        run = RunModel.query.filter_by(id=id).first()

        if run:
            run = run_schema.dump(run).data

            return {'status': 'success', 'data': run}, 200
        else:
            return {'status': 'failure', 'data': None}, 404

    def delete(self, id):
        run = RunModel.query.filter_by(id=id).delete()

        db.session.commit()

        result = run_schema.dump(run).data

        return {"status": 'success', 'data': result}, 204


class RunFinished(Resource):
    def patch(self, id):
        run = RunModel.query.filter_by(id=id).first()

        if not run:
            return {'message': 'Run does not exist'}, 400

        run.end_date = db.func.current_timestamp()
        db.session.commit()

        result = run_schema.dump(run).data

        return {"status": 'success', 'data': result}, 204
