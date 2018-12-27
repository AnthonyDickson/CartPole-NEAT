from flask_migrate import Migrate, MigrateCommand
from flask_script import Manager

from api.model import db
from api.run import create_app

app = create_app('api.config')

migrate = Migrate(app, db)
manager = Manager(app)
manager.add_command('db', MigrateCommand)

if __name__ == '__main__':
    manager.run()
