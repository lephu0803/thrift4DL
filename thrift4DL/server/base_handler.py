# Copyright (c) 2019 congvm
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from .connectors import Receiver, Deliver
import multiprocessing
from .ttypes import TVisionResult
from thrift.Thrift import TType, TMessageType, TApplicationException
import traceback
from queue import Empty
import numpy as np

IDLE_QUEUE_BLOCK_TIME_SEC = 10


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
    def __init__(self, model_path, gpu_id, mem_fraction, client_queue, batch_group_timeout, batch_infer_size):
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
        self._pid = np.random.randint(1000)

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

    def error_handle(self, args_dict):
        args_dict['msg_type'] = TMessageType.REPLY
        args_dict['result'].success = TVisionResult(error_code=-1, response="")
        return args_dict

    def success_handle(self, result, args_dict):
        args_dict['result'] = result
        args_dict['msg_type'] = TMessageType.REPLY
        return args_dict

    def model_process(self, model, image_binary):
        img_arr = self.preprocessing(model, image_binary)
        pred_result = self.predict(model, img_arr)
        pred_result = self.postprocessing(model, pred_result)
        return pred_result

    def run(self):
        env_params = self.get_env(self.gpu_id, self.mem_fraction)
        model = self.get_model(self.model_path, env_params)
        while True:
            client = self.client_queue.get()
            args_dict = self.receiver.process(client)
            image_binary = args_dict['image_binary']
            result = args_dict['result']
            try:
                pred_result = self.model_process(model, image_binary)
                assert isinstance(pred_result, str), ValueError(
                    "Expected result to be a string")
                args_dict['result'].success = TVisionResult(
                    error_code=0, response=pred_result)
                args_dict = self.success_handle(result=result,
                                                args_dict=args_dict)
            except Exception as e:
                print(traceback.format_exc())
                args_dict = self.error_handle(args_dict)
            try:
                self.deliver.process(args_dict)
            except Exception as e:
                print(traceback.format_exc())

class BatchingHandler(Handler):

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
                    args_dict = self.receiver.process(client)
                    if args_dict is None:
                        continue
                    batch_input.append(args_dict)
                    timeout = self.batch_group_timeout
                except Empty:
                    is_empty = True

                if (len(batch_input) >= self.batch_infer_size) or (is_empty and len(batch_input) > 0):
                    is_done = True
                    print(f"Process: {self._pid}:", len(batch_input))
                    yield batch_input
            except Exception as e:
                print(traceback.format_exc())

    def run(self):
        env_params = self.get_env(self.gpu_id, self.mem_fraction)
        model = self.get_model(self.model_path, env_params)
        for batch_input in self.get_batch():
            import time

            if len(batch_input) > 0:
                batch_image_binary = []
                batch_pred_result = []
                batch_final_result = []
                for connection_info in batch_input:
                    try:
                        image_binary = connection_info['image_binary']
                        result = connection_info['result']
                        image_binary = self.preprocessing(model, image_binary)
                        batch_image_binary.append(image_binary)
                    except Exception as e:
                        print(traceback.format_exc())
                        connection_info = self.error_handle(connection_info)
                        self.deliver.process(connection_info)
                        continue
                try:
                    batch_pred_result = self.predict(model, batch_image_binary)
                except Exception as e:
                    print(traceback.format_exc())

                for pred_result in batch_pred_result:
                    batch_final_result.append(
                        self.postprocessing(model, pred_result))

                for connection_info, pred_result in zip(batch_input, batch_final_result):
                    try:
                        assert isinstance(pred_result, str), ValueError(
                            "Expected result to be a string")
                        result.success = TVisionResult(error_code=0,
                                                       response=pred_result)
                        connection_info = self.success_handle(result=result,
                                                              args_dict=connection_info)
                    except Exception as e:
                        print(traceback.format_exc())
                        connection_info = self.error_handle(connection_info)
                        
                    try:
                        self.deliver.process(connection_info)
                    except Exception as e:
                        print(traceback.format_exc())