import os
from keras.layers import Conv2D, Input,MaxPool2D, Reshape,Activation,Flatten, Dense, Permute
from keras.models import Model, Sequential
import tensorflow as tf
from keras.layers.advanced_activations import PReLU
import numpy as np
from thirdparty import tools_matrix
import cv2

class SSDFaceDetector(object):

    def __init__(self, ssd_model_path, onet_model_path, ssd_detection_thres = 0.3, 
                 onet_thres = 0.7, face_label = 1):
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.config.log_device_placement = False
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=self.config)
        
        with self.graph.as_default():
            with self.sess.as_default():
                self.__load_model(ssd_model_path)
                
        self.onet = self.__create_Kao_Onet(onet_model_path)
        self.ssd_detection_thres = ssd_detection_thres
        self.onet_thres = onet_thres
        self.face_label = face_label
        self.onet_input_size = 48
    

    def __load_model(self, model_path):
        model_exp = os.path.expanduser(model_path)
        print('Model filename: %s' % model_exp)
        with tf.gfile.GFile(model_exp, 'rb') as fid:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(fid.read())
            tf.import_graph_def(graph_def, name='')
 
        
    def __create_Kao_Onet(self, weight_path = None):
        input = Input(shape = [48,48,3])
        x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv1')(input)
        x = PReLU(shared_axes=[1,2],name='prelu1')(x)
        x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)
        x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv2')(x)
        x = PReLU(shared_axes=[1,2],name='prelu2')(x)
        x = MaxPool2D(pool_size=3, strides=2)(x)
        x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv3')(x)
        x = PReLU(shared_axes=[1,2],name='prelu3')(x)
        x = MaxPool2D(pool_size=2)(x)
        x = Conv2D(128, (2, 2), strides=1, padding='valid', name='conv4')(x)
        x = PReLU(shared_axes=[1,2],name='prelu4')(x)
        x = Permute((3,2,1))(x)
        x = Flatten()(x)
        x = Dense(256, name='conv5') (x)
        x = PReLU(name='prelu5')(x)

        classifier = Dense(2, activation='softmax',name='conv6-1')(x)
        bbox_regress = Dense(4,name='conv6-2')(x)
        landmark_regress = Dense(10,name='conv6-3')(x)
        model = Model([input], [classifier, bbox_regress, landmark_regress])
    
        if weight_path:
            model.load_weights(weight_path, by_name=True)

        return model 
    
    def __norm_coord(self, x,max_value):
        return int(min(max(x,0),max_value))

    def ssd_predict(self, img_expanded):
        image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
        bboxes_tensor = self.graph.get_tensor_by_name('detection_boxes:0')
        scores_tensor = self.graph.get_tensor_by_name('detection_scores:0')
        classes_tensor = self.graph.get_tensor_by_name('detection_classes:0')
        nofaces_tensor = self.graph.get_tensor_by_name('num_detections:0')
        
        feed_dict={image_tensor: img_expanded}
        return self.sess.run([bboxes_tensor, scores_tensor, classes_tensor, nofaces_tensor], 
                             feed_dict=feed_dict)
    
    def mtcnn_predict(self, face_batch, original_bboxes, original_size):
        imgw, imgh = original_size
        face_batch_normed = (face_batch.copy() - 127.5) / 127.5
        classifier, bboxes_regression, landmark_regression = self.onet.predict(face_batch_normed)
        
        

        rectangles = tools_matrix.filter_face_48net(classifier, bboxes_regression, 
                                                    landmark_regression, original_bboxes, imgw, imgh,
                                                    self.onet_thres)
        
        bboxes, landmarks = self.__convert_mtccn_bboxes(rectangles)
        return np.array(bboxes), np.array(landmarks).T
    

    def __convert_mtccn_bboxes(self, rectangles):
        bboxes = []
        landmarks = []
        for rec in rectangles:
            bboxes.append(rec[:5])
            pts0,pts1,pts2,pts3,pts4,pts5,pts6,pts7,pts8,pts9 = rec[5:]
            landmarks.append([pts0, pts2, pts4, pts6, pts8, pts1, pts3, pts5, pts7, pts9])
        return bboxes, landmarks
            
    
    def detect(self, image_list):
        result = []
        image_batch = []
        for img in image_list:
            img = cv2.resize(img, (320,320))
            image_batch.append(img[np.newaxis,:,:,:])
        image_batch = np.concatenate(image_batch)
            
        ssd_bboxes, ssd_scores, ssd_classes, ssd_nofaces = self.ssd_predict(image_batch)
        for bboxes, scores, classes, img in zip(ssd_bboxes, ssd_scores, ssd_classes, image_list):
            imgh, imgw, _ = img.shape
            mtcnn_bboxes = []
            face_images = []
            for bbox, score, cl in zip(bboxes, scores, classes):
                if score >= self.ssd_detection_thres and cl == self.face_label:
                    ymin, xmin, ymax, xmax = bbox
                    w, h = xmax - xmin, ymax - ymin
                    wm, hm = 0, 0
                    xmin  = self.__norm_coord(int((xmin - wm) * imgw), imgw)
                    xmax  = self.__norm_coord(int((xmax + wm) * imgw), imgw)
                    ymin  = self.__norm_coord(int((ymin - hm) * imgh), imgh)
                    ymax  = self.__norm_coord(int((ymax + hm) * imgh), imgh)
                    mtcnn_bboxes.append([xmin,ymin,xmax,ymax,score])
                    
                    if xmax - xmin > 0 and ymax - ymin > 0:
                        crop_img = img[ymin:ymax, xmin:xmax]
                        face_img = cv2.resize(crop_img, (self.onet_input_size, self.onet_input_size))
                        face_images.append(np.array(face_img)[np.newaxis,:,:,:])
            
            if len(face_images) > 0:
                face_batch = np.concatenate(face_images)
                bboxes, landmarks = self.mtcnn_predict(face_batch, mtcnn_bboxes, (imgw, imgh))
                result.append((bboxes, landmarks))
            else:
                result.append(([], []))
        return result