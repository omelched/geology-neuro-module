from flask_restful import Resource
import bson

import neuroAPI.utils
from neuroAPI.database.models import Deposit


class DepositResource(Resource):
    def get(self, _id='000000000000000000000000'):  # noqa
        if _id == '000000000000000000000000':
            return 'No parameter set', 404

        try:
            oId = bson.ObjectId(_id)
        except (TypeError, bson.errors.InvalidId) as e:
            return str(e), 404
        except Exception as e:
            neuroAPI.utils.handle_exception(neuroAPI.utils.logger, e)
            return 'Shit on my side', 500

        deposit = Deposit.objects(id=oId)
        if deposit:
            return deposit[0].to_json(), 200
        else:
            return "No match", 400

    # def post
