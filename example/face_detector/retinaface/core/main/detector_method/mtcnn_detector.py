from thirdparty import detect_face
from core.entry.face import Face
import tensorflow as tf
from core.processing.face_preprocessor import FacePreprocessor
from core.base.logging_model import logger
from core.utils import common

class MTCNNFaceDetector:
    min_face_size = 15
    threshold = [0.6, 0.7, 0.7]
    scale_factor = 0.709
    image_scale_factor = 4

    def __init__(self, mtcnn_model_path,
                 face_crop_margin = 44,
                 image_size = [112,112],
                 gpu_memory_fraction = 0.3):
        self.face_crop_margin = face_crop_margin
        self.image_size = image_size
        self.gpu_memory_fraction = gpu_memory_fraction
        self.face_preprocessor = FacePreprocessor(self.face_crop_margin, self.image_size)
        self.pnet, self.rnet, self.onet = self.__setup_mtcnn(mtcnn_model_path)

    def __setup_mtcnn(self, mtcnn_model_path):
        with tf.Graph().as_default():
            # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = self.gpu_memory_fraction)
            gpu_options = tf.GPUOptions(allow_growth = True)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                return detect_face.create_mtcnn(sess, mtcnn_model_path)

    def detect(self, image, request_id):
        faces = []
        bounding_boxes, landmarks = detect_face.detect_face(image, self.min_face_size, self.pnet,
                                                            self.rnet, self.onet, self.threshold, self.scale_factor)
                                                            
        nrof_faces = len(bounding_boxes)
        # logger.info("[FaceId] Number of face in request_id {} is {}".format(request_id, nrof_faces))

        for i in range(nrof_faces):
            if bounding_boxes[i][4] > 0.95:
                face = Face()
                face.request_id = request_id
                face.bounding_box = bounding_boxes[i, 0:4]
                face.landmarks = landmarks[:, i]
                face.aligned_image = self.face_preprocessor.process(image, face.bounding_box, face.landmarks)
                
                if face.aligned_image is not None:
                    faces.append(face)
                else:
                    logger.info("[FaceId] Face {}/{} of request_id {} with bbox {} is too small size or high angle ".format(i, nrof_faces, request_id, bounding_boxes[i, 0:4]))
        return faces

    def detect_batch(self, urls, request_ids):
        faces = []
        for i in range(len(urls)):
            # img = common.load_image_url(urls[i])
            img = common.read_image(urls[i])
            if img is not None:
                faces += self.detect(img, request_ids[i])
        return faces 
