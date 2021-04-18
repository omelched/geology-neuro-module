# from uuid import UUID
# from typing import Union, List
# from abc import abstractmethod
#
# import torch
#
# from neuroAPI.database.models import NeuralModelMetrics, MetricType, Rock
# from neuroAPI.neuralmodule.network import NeuralNetwork
#
# _METRIC_ID_CACHE: dict[str, UUID] = {}  # to buffer metric.name -> metric.id mapping
#
#
# class BaseMetric(NeuralModelMetrics):
#
#     def __init__(self, neural_network: NeuralNetwork, epoch: int, **kwargs):  # TODO: 3 AM shitcoding. refactor
#         if not isinstance(neural_network, NeuralNetwork):
#             raise ValueError
#
#         if isinstance(kwargs['name'], str) and isinstance(kwargs['mtype'], MetricType):
#             _name = kwargs['name']
#             mtype = kwargs['mtype']
#         else:
#             raise ValueError  # TODO: log + sys.exc_info()[0]
#
#         if 'value' in kwargs.keys():
#             self.value = kwargs['value']
#         elif all(k in kwargs for k in ('true', 'pred')):
#             self.value = self.calculate(kwargs['true'], kwargs['pred'])
#         else:
#             raise NotImplementedError
#
#         self.epoch = int(epoch)  # TODO: catch exeptions
#
#         if not self.name in _METRIC_ID_CACHE.keys():
#             metric_id = self.get_create_metric(_name, mtype)
#             _METRIC_ID_CACHE[_name] = metric_id
#         else:
#             metric_id = _METRIC_ID_CACHE[_name]
#
#         self.metric_id = metric_id
#         self.neural_model_id = neural_network.id
#
#     def calculate(self, true: torch.Tensor, pred: torch.Tensor) -> Union[float, List[List[int]]]:
#         if not isinstance(true, torch.Tensor) or not isinstance(pred, torch.Tensor):
#             raise ValueError
#
#         return self._calculate(true, pred)
#
#     @staticmethod
#     @abstractmethod
#     def _calculate(true: torch.Tensor, pred: torch.Tensor) -> Union[float, List[List[int]]]:
#         raise NotImplementedError
#
#
# class PYCMClassStatMetric(BaseMetric):
#
#     def __init__(self, neural_model: NeuralNetwork, epoch: int, rock: Rock, value: Union[str, float, int], name: str):
#         super(PYCMClassStatMetric, self).__init__(neural_model, epoch, value=value, name=name,
#                                                   type=MetricType.class_stat)
#         self.rock_id = rock.id
#
#     @staticmethod
#     def _calculate(true: torch.Tensor, pred: torch.Tensor) -> Union[float, List[List[int]]]:
#         raise NotImplementedError
#
#
# class PYCMOverallStatMetric(BaseMetric):
#
#     def __init__(self, neural_model: NeuralNetwork, epoch: int, value: Union[str, float, int], name:str):
#         super(PYCMOverallStatMetric, self).__init__(neural_model, epoch, value=value, name=name,
#                                                     type=MetricType.overall_stat)
#
#     @staticmethod
#     def _calculate(true: torch.Tensor, pred: torch.Tensor) -> Union[float, List[List[int]]]:
#         raise NotImplementedError
#
#
# def parse_PYCM(pycm_dict_class: dict[str, dict[str, Union[int, float, str]]],
#                pycm_dict_overall: Union[str, int, float, list[float]],
#                neural_model: NeuralNetwork, epoch: int)\
#         -> List[Union[PYCMOverallStatMetric, PYCMClassStatMetric]]:
#     for k, v in pycm_dict_class.items():
#         for kk, vv in v.items():
#             metric = PYCMClassStatMetric(neural_model,
#                                          epoch,
#                                          )