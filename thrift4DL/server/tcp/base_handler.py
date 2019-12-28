# Copyright (c) 2019 congvm
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from .connectors import Receiver, Deliver, Validator
import multiprocessing
from .ttypes import TVisionResult
from thrift.Thrift import TType, TMessageType, TApplicationException
import traceback
from queue import Empty
import numpy as np

IDLE_QUEUE_BLOCK_TIME_SEC = 10
ERROR_MESSAGE = 'Internal Error'
SUCCESS_MESSAGE = 'Successful'
ERROR_CODE = -1
ERROR_RESPONSE = ""


class BaseHandler():
    def __init__(self, model_path, gpu_id, mem_fraction, client_queue, batch_group_timeout, batch_infer_size):
        pass

    def get_env(self, gpu_id, mem_fraction):
        raise NotImplementedError

    def get_model(self, model_path, env_params):
        raise NotImplementedError

    def preprocessing(self, model, input):
        raise NotImplementedError

    def postprocessing(self, model, input):
        raise NotImplementedError

    def predict(self, model, input):
        raise NotImplementedError

    def error_handle(self, args_dict):
        raise NotImplementedError

    def success_handle(self, result, args_dict):
        raise NotImplementedError

    def model_process(self, model, image_binary):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError


class Handler(multiprocessing.Process):
    def __init__(self, model_path, gpu_id, mem_fraction,
                 client_queue, batch_group_timeout, batch_infer_size):
        multiprocessing.Process.__init__(self)
        print("Init Handler")
        self.client_queue = client_queue
        self.gpu_id = gpu_id
        self.mem_fraction = mem_fraction
        self.model_path = model_path
        self.batch_infer_size = batch_infer_size
        self.batch_group_timeout = self._milisec_to_sec(batch_group_timeout)
        self.receiver = Receiver()
        self.deliver = Deliver()
        self.validator = Validator()
        self._pid = np.random.randint(1000)
        self.processing_name = ['predict']

    def _milisec_to_sec(self, sec):
        return sec/1000

    def get_env(self, gpu_id, mem_fraction):
        env_params = None
        return env_params

    def get_model(self, model_path, env_params):
        model = None
        return model

    def preprocessing(self, model, input):
        raise NotImplementedError

    def postprocessing(self, model, input):
        raise NotImplementedError

    def predict(self, model, input):
        raise NotImplementedError

    def _predict_error_handle(self, connection_info):
        connection_info['result'].success = TVisionResult(
            ERROR_CODE, ERROR_MESSAGE, ERROR_RESPONSE)
        return connection_info

    def _predict_success_handle(self, response, connection_info):
        connection_info['result'].success = TVisionResult(
            0, SUCCESS_MESSAGE, response)
        return connection_info

    def _model_process(self, model, image_binary):
        img_arr = self.preprocessing(model, image_binary)
        pred_result = self.predict(model, img_arr)
        pred_result = self.postprocessing(model, pred_result)
        return pred_result

    def run(self):
        env_params = self.get_env(self.gpu_id, self.mem_fraction)
        model = self.get_model(self.model_path, env_params)
        while True:
            client = self.client_queue.get()
            connection_info = self.receiver.process(client)
            connection_info = self.validator.process(connection_info)
            image_binary = connection_info['image_binary']
            try:
                pred_response = self._model_process(model, image_binary)
                assert isinstance(pred_response, str), ValueError(
                    "Expected result to be a string")
                connection_info = self._predict_success_handle(pred_response,
                                                               connection_info)
            except Exception as e:
                print(traceback.format_exc())
                connection_info = self._predict_error_handle(connection_info)
            self.deliver.process(connection_info)


class VisionHandler(Handler):

    def get_batch(self):
        """ Block queue for a while to wait incomming request
        """
        batch_input = []
        is_done = False
        is_empty = False
        timeout = IDLE_QUEUE_BLOCK_TIME_SEC
        while True:
            try:
                if is_done:
                    batch_input.clear()
                    is_done = False
                    is_empty = False
                    timeout = IDLE_QUEUE_BLOCK_TIME_SEC
                try:
                    client = self.client_queue.get(block=True,
                                                   timeout=timeout)
                    connection_info = self.receiver.process(client)
                    connection_info = self.validator.process(connection_info)
                    if connection_info['name'] not in self.processing_name:
                        self.deliver.process(connection_info)
                        continue
                    batch_input.append(connection_info)
                    timeout = self.batch_group_timeout
                except Empty:
                    is_empty = True
                if (len(batch_input) >= self.batch_infer_size) or (is_empty and len(batch_input) > 0):
                    is_done = True
                    print(f"Process: {self._pid}:", len(batch_input))
                    yield batch_input
            except Exception as e:
                print(traceback.format_exc())

    def _get_default_batch_pred_result(self, batch_len):
        return [TVisionResult(error_code=-1,
                              error_message="",
                              response="")]*batch_len

    def run(self):
        env_params = self.get_env(self.gpu_id, self.mem_fraction)
        model = self.get_model(self.model_path, env_params)
        for batch_connection_info in self.get_batch():
            if len(batch_connection_info) > 0:
                batch_image_binary = []
                batch_pred_result = self._get_default_batch_pred_result(
                    len(batch_connection_info))

                for connection_info in batch_connection_info:
                    try:
                        image_binary = connection_info['image_binary']
                        image_binary = self.preprocessing(model, image_binary)
                        batch_image_binary.append(image_binary)
                    except Exception as e:
                        print(traceback.format_exc())
                        connection_info = self._predict_error_handle(
                            connection_info)
                        self.deliver.process(connection_info)

                # Only process if there is at least one decoded image 
                if len(batch_image_binary) > 0:
                    try:
                        batch_pred_result = self.predict(model, batch_image_binary)
                    except Exception as e:
                        print(traceback.format_exc())

                    for connection_info, pred_result in zip(batch_connection_info, batch_pred_result):
                        try:
                            pred_result = self.postprocessing(model, pred_result)
                            connection_info = self._predict_success_handle(pred_result,
                                                                        connection_info)
                        except Exception as e:
                            print(traceback.format_exc())
                            connection_info = self._predict_error_handle(
                                connection_info)
                        self.deliver.process(connection_info)
