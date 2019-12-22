
import multiprocessing
from ..ttypes import TResult
from thrift.Thrift import TType, TMessageType, TApplicationException
import traceback
from queue import Empty


class BaseHandler(multiprocessing.Process):
    def __init__(self, model_path, gpu_id, mem_fraction, args_queue, result_queue, batch_group_timeout, batch_infer_size):
        multiprocessing.Process.__init__(self)
        print("Init Handler")
        self.result_queue = result_queue
        self.args_queue = args_queue
        self.gpu_id = gpu_id
        self.mem_fraction = mem_fraction
        self.model_path = model_path
        self.batch_infer_size = batch_infer_size
        self.batch_group_timeout = batch_group_timeout

    def get_env(self, gpu_id, mem_fraction):
        raise NotImplementedError

    def get_model(self, model_path, gpu_id, mem_fraction):
        raise NotImplementedError

    def preprocessing(self, input):
        raise NotImplementedError

    def postprocessing(self, input):
        raise NotImplementedError

    def predict(self, model, input):
        raise NotImplementedError

    def send_response(self, args_dict):
        self.result_queue.put(args_dict)

    def error_handler(self, args_dict):
        args_dict['msg_type'] = TMessageType.EXCEPTION
        args_dict['result'] = TApplicationException(
            TApplicationException.INTERNAL_ERROR, 'Internal error')
        return args_dict

    def run(self):
        model = self.get_model(self.model_path, self.gpu_id, self.mem_fraction)
        while True:
            args_dict = self.args_queue.get()
            try:
                args_request = args_dict['args_request']
                result = args_dict['result']
                args = self.preprocessing(args_request)
                pred_result = self.predict(model, args)
                pred_result = self.postprocessing(pred_result)
                assert isinstance(pred_result, str), ValueError(
                    "Expected result to be a string")
                result.success = TResult(error_code=0, response=pred_result)
                args_dict['result'] = result
                args_dict['msg_type'] = TMessageType.REPLY
            except Exception as e:
                print(traceback.format_exc())
                args_dict = self.error_handler(args_dict)
            self.send_response(args_dict)


class BatchingBaseHandler(BaseHandler):
    def get_batch(self):
        batch_input = []
        is_done = False
        while True:
            try:
                args_dict = self.args_queue.get(
                    block=False, timeout=self.batch_group_timeout)
                batch_input.append(args_dict)
            except Empty:
                is_done = True

            if len(batch_input) > 0 and (len(batch_input) >= self.batch_infer_size or is_done):
                yield batch_input
                batch_input.clear()
                is_done = False

    def run(self):
        model = self.get_model(self.model_path, self.gpu_id, self.mem_fraction)
        # while True:
        for batch_input in self.get_batch():
            batch_args_request = []
            batch_pred_result = batch_args_request.copy()
            for args_dict in batch_input:
                try:
                    args_request = args_dict['args_request']
                    result = args_dict['result']
                    args = self.preprocessing(args_request)
                except Exception as e:
                    print(traceback.format_exc())
                    args_dict = self.error_handler(args_dict)
                    self.send_response(args_dict)
            try:
                batch_args_request.append(args) 
                batch_pred_result = self.predict(model, batch_args_request)
            except Exception as e:
                print(traceback.format_exc())

            for pred_result in batch_pred_result:
                try:
                    pred_result = self.postprocessing(pred_result)
                    assert isinstance(pred_result, str), ValueError(
                        "Expected result to be a string")
                    result.success = TResult(error_code=0, 
                                            response=pred_result)
                    args_dict['result'] = result
                    args_dict['msg_type'] = TMessageType.REPLY
                    self.send_response(args_dict)
                except Exception as e:
                    print(traceback.format_exc())
                    args_dict = self.error_handler(args_dict)
                    self.send_response(args_dict)
