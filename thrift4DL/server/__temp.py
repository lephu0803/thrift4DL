
# class Ventilator(multiprocessing.Process):
#     def __init__(self, client_queue, connection_queue, batch_infer_size=1, batch_group_timeout=10):
#         print("Init Ventilator")
#         multiprocessing.Process.__init__(self)
#         self.client_queue = client_queue
#         self.connection_queue = connection_queue
#         self.batch_infer_size = batch_infer_size
#         self.batch_group_timeout = self._milisec_to_sec(batch_group_timeout)
#         # self.n_process = len(self.client_queues)
#         self.print_server_info()

#     def print_server_info(self):
#         import pprint
#         pp = pprint.PrettyPrinter(indent=3)
#         print("="*50)
#         print("Server Information:\n")
#         pp.pprint(self.__dict__)
#         print("="*50)

#     def _milisec_to_sec(self, sec):
#         return sec/1000

#     def get_batch(self):
#         """ Block queue for a while to wait incomming request
#         """
#         batch_clients = []
#         is_done = False
#         is_empty = False
#         timeout = IDLE_QUEUE_BLOCK_TIME_SEC
#         while True:
#             try:
#                 if is_done:
#                     # Reset state
#                     batch_clients.clear()
#                     is_done = False
#                     is_empty = False
#                     timeout = IDLE_QUEUE_BLOCK_TIME_SEC
#                 try:
#                     client = self.connection_queue.get(block=True,
#                                                        timeout=timeout)
#                     batch_clients.append(client)
#                     timeout = self.batch_group_timeout
#                 except multiprocessing.queues.Empty:
#                     is_empty = True

#                 if (len(batch_clients) >= self.batch_infer_size) or (is_empty and len(batch_clients) > 0):
#                     is_done = True
#                     yield batch_clients

#             except Exception as e:
#                 print(traceback.format_exc())

#     def run(self):
#         for batch_client in self.get_batch():
#             # Distribute
#             print("Ventilator:", batch_client)
#             # client_queue_idx = np.random.randint(self.n_process)
#             # self.client_queues[client_queue_idx].put(batch_client)
#             self.client_queue.put(batch_client)
#             print(self.client_queue.qsize())


# class TModelPoolServerV4():
#     def __init__(self, host, port, handler_cls,
#                  model_path, gpu_ids, mem_fractions,
#                  batch_infer_size=1, batch_group_timeout=10, verbose=True, logger=None):
#         self.handler_cls = handler_cls
#         self.model_path = model_path
#         self.gpu_ids = gpu_ids
#         self.mem_fractions = mem_fractions
#         self.host = host
#         self.port = port
#         self.socket = TSocket.TServerSocket(host=self.host, port=self.port)
#         self.client_queue = self.get_mpqueue()
#         self.connection_queue = multiprocessing.Queue()
#         self.batch_infer_size = batch_infer_size
#         self.batch_group_timeout = batch_group_timeout
#         self.handlers = []
#         self.is_running = False
#         self.verbose = verbose

#     def print_server_info(self):
#         import pprint
#         pp = pprint.PrettyPrinter(indent=3)
#         print("="*50)
#         print("Server Information:\n")
#         pp.pprint(self.__dict__)
#         print("="*50)

#     def get_mpqueue(self):
#         # q = Manager().Queue()
#         # q = multiprocessing.JoinableQueue()
#         q = multiprocessing.Queue()
#         return q

#     def prepare(self):
#         for i in range(len(self.gpu_ids)):
#             # client_queue = self.get_mpqueue()
#             # self.client_queues.append(client_queue)
#             wrk = self.handler_cls(model_path=self.model_path,
#                                    gpu_id=self.gpu_ids[i],
#                                    mem_fraction=self.mem_fractions[i],
#                                    client_queue=self.client_queue,
#                                    batch_infer_size=self.batch_infer_size,
#                                    batch_group_timeout=self.batch_group_timeout)

#             wrk.daemon = True
#             wrk.start()
#             self.handlers.append(wrk)

#         ven_wrk = Ventilator(self.client_queue, self.connection_queue,
#                              batch_infer_size=self.batch_infer_size,
#                              batch_group_timeout=self.batch_group_timeout)
#         ven_wrk.daemon = True
#         ven_wrk.start()
#         # ven_wrk.join()
#         self.is_running = True

#         if self.verbose:
#             self.print_server_info()

#     def serve(self):
#         self.prepare()
#         self.socket.listen()
#         print("Service started")
#         while self.is_running:
#             try:
#                 client = self.socket.accept()
#                 if not client:
#                     continue
#                 self.connection_queue.put(client)
#             except (SystemExit, KeyboardInterrupt):
#                 break
#             except Exception as err:
#                 tb = traceback.format_exc()
#                 print(tb)