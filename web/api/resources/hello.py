from flask_restful import Resource


class Hello(Resource):
    def get(self, name='World'):
        return {"message": "Hello, %s!" % name.capitalize()}
