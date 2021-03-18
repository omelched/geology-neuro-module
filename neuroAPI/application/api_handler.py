from flask_restful import Api, Resource, reqparse

from neuroAPI.application import server
from neuroAPI.application.resources import DepositResource


class _NeuroAPI(Api):
    pass


api = _NeuroAPI(server)
api.add_resource(DepositResource, '/deposit/<string:_id>', )
