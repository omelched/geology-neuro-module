from flask_restful import Resource

import neuroAPI.utils
from neuroAPI.database.models import Deposit


class DepositResource(Resource):
    def get(self):  # noqa

        return 501

