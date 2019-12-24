import cv2
import numpy as np
from ..utils import constant as cons
from ..utils import common


class FacePyramidPadding():

    def __init__(self):
        self.is_pyramid_image = False
        self.pyramid_image_scope = None
        self.pyramid_image_down_scale = [1, 2, 4]
        self.original_image = None
        self.resized_image = None
        self.half_resized_image = None
        self.image_pyramid_scale = 1
        self.image_pyramid_second_scale = 1
        self.image_normal_scale = 1

    def preprocessing(self, image):
        image_shape = image.shape
        if image_shape[0] <= cons.PYRAMID_IMAGE_THRESHOLD[0] and image_shape[1] <= cons.PYRAMID_IMAGE_THRESHOLD[1]:
            self.is_pyramid_image = True
            self.image_normal_scale = 1
            self.resized_image, self.half_resized_image, self.image_pyramid_scale = self.__get_pyramid_image(
                image)
            self.pyramid_image_scope = self.__get_image_scope_areas()
            pyramid_combined_image = self.__combine_pyramid_image()
            pyramid_combined_image = self.__scale_combine_pyramid_image(
                pyramid_combined_image)
            padding_image = self.pad_image(
                pyramid_combined_image, cons.SCALES_FIRST)
        else:
            self.is_pyramid_image = False
            self.image_pyramid_scale = 1
            self.image_normal_scale = common.calculate_image_scale_ratio(
                image, cons.SCALES_FIRST)
            resized_image = common.resize_image(image, self.image_normal_scale)
            padding_image = self.pad_image(resized_image, cons.SCALES_FIRST)
        return padding_image, self.image_normal_scale

    def postprocessing(self, bboxes, landmarks):
        areas_index, mapping_bboxes, mapping_landmarks = self.__map_original_space(
            bboxes, landmarks)

        sorted_idx = []
        sorted_bboxes = []
        sorted_landmarks = []
        for idx, bbox, lmk in sorted(zip(areas_index, mapping_bboxes, mapping_landmarks),  key=lambda pair: pair[0]):
            sorted_idx.append(idx)
            sorted_bboxes.append(bbox)
            sorted_landmarks.append(lmk)

        sorted_bboxes = np.asarray(sorted_bboxes)
        sorted_landmarks = np.asarray(sorted_landmarks)
        keep_idxs = self.__non_max_suppression(
            sorted_bboxes, cons.IOU_THRESHOLD)

        return sorted_bboxes[keep_idxs], sorted_landmarks[keep_idxs]

    def __map_original_space(self, bboxes, landmarks):
        areas_index = []
        mapping_bboxes = []
        mapping_landmarks = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            lmk = landmarks[i]
            area_index, mapping_bbox, mapping_lmk = self.__map_prediction_with_original_image(
                self.pyramid_image_scope, self.image_pyramid_scale*self.image_pyramid_second_scale, bbox, lmk)
            areas_index.append(area_index)
            mapping_bboxes.append(mapping_bbox)
            mapping_landmarks.append(mapping_lmk)

        return areas_index, mapping_bboxes, mapping_landmarks

    def pad_image(self, resized_img, desired_size):
        resized_img_size = resized_img.shape[:2]
        delta_w = desired_size[1] - resized_img_size[1]
        delta_h = desired_size[0] - resized_img_size[0]
        top, bottom = 0, delta_h
        left, right = 0, delta_w
        color = [0, 0, 0]
        padding_img = cv2.copyMakeBorder(
            resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return padding_img

    def __hconcat_resize_min(self, image_list, interpolation=cv2.INTER_LINEAR):
        h_max = max(img.shape[1] for img in image_list)
        image_list_resize = []
        for img in image_list:
            resized_img = cv2.resize(
                img, (int(img.shape[1] * h_max / img.shape[0]), h_max), interpolation=interpolation)
            image_list_resize.append(resized_img)
        return cv2.hconcat(image_list_resize)

    def __vconcat_resize_max(self, image_list, interpolation=cv2.INTER_LINEAR):
        w_max = max(img.shape[0] for img in image_list)
        image_list_resize = []
        for img in image_list:
            resized_img = cv2.resize(img, (w_max, int(
                img.shape[0] * w_max / img.shape[1])), interpolation=interpolation)
            image_list_resize.append(resized_img)
        return cv2.vconcat(image_list_resize)

    def __concat_tile_resize(self, image_list_2d, interpolation=cv2.INTER_LINEAR):
        image_list_v = [self.__vconcat_resize_max(
            image_list_h, interpolation=cv2.INTER_LINEAR) for image_list_h in image_list_2d]
        return self.__hconcat_resize_min(image_list_v, interpolation=cv2.INTER_LINEAR)

    def __get_pyramid_image(self, image):
        image_scale = common.calculate_image_scale_ratio(
            image, cons.SCALES_FIRST)
        resized_image = common.resize_image(image, image_scale)
        half_resized_image = common.resize_image(image, image_scale/2)
        return resized_image, half_resized_image, image_scale

    def __combine_pyramid_image(self):
        tmp = self.half_resized_image.copy()
        tmp[:, :, :] = 0
        quarter_combine_image = self.__concat_tile_resize(
            [[self.half_resized_image, tmp], [tmp, tmp]])
        pyramid_combined_image = self.__concat_tile_resize(
            [[self.resized_image], [self.half_resized_image, quarter_combine_image]])
        return pyramid_combined_image

    def __scale_combine_pyramid_image(self, pyramid_combined_image):
        self.image_pyramid_second_scale = 1
        if pyramid_combined_image.shape[1] > cons.SCALES_FIRST[1]:
            self.image_pyramid_second_scale = cons.SCALES_FIRST[1] / \
                pyramid_combined_image.shape[1]
            pyramid_combined_image = cv2.resize(pyramid_combined_image, None, None, fx=self.image_pyramid_second_scale,
                                                fy=self.image_pyramid_second_scale, interpolation=cv2.INTER_LINEAR)
            self.pyramid_image_scope *= self.image_pyramid_second_scale
        return pyramid_combined_image

    def __get_image_scope_areas(self):
        scope_biggest_image = [
            0, 0, self.resized_image.shape[1], self.resized_image.shape[0]]
        x = scope_biggest_image[0] + scope_biggest_image[2]
        y = 0
        scope_medium_image = [
            x, y, x + self.half_resized_image.shape[1], y + self.half_resized_image.shape[0]]
        x = scope_biggest_image[0] + scope_biggest_image[2]
        y = scope_medium_image[1] + scope_medium_image[3]
        scope_smallest_image = [
            x, y, x + np.ceil(self.half_resized_image.shape[1]/2), y + np.ceil(self.half_resized_image.shape[0]/2)]
        image_areas = np.asarray(
            [scope_biggest_image, scope_medium_image, scope_smallest_image])
        return image_areas

    def __is_point_in_rectangle(self, area, point):
        x1, y1, x2,  y2 = area
        x, y = point
        if (x > x1 and x < x2 and
                y > y1 and y < y2):
            return True
        else:
            return False

    def __get_face_areas(self, areas, lmk):
        for idx, area in enumerate(areas):
            if self.__is_point_in_rectangle(area, lmk[0]) and self.__is_point_in_rectangle(area, lmk[4]):
                return idx

    def __map_prediction_with_original_image(self, areas, down_scale, bbox, lmk):
        area_idx = self.__get_face_areas(areas, lmk)
        up_scale = self.pyramid_image_down_scale
        translation_bbox = bbox.copy()
        translation_bbox[::2][:2] = (
            bbox[::2][:2] - areas[area_idx][0])*up_scale[area_idx]/down_scale
        translation_bbox[1::2] = (
            bbox[1::2] - areas[area_idx][1])*up_scale[area_idx]/down_scale

        translation_lmk = lmk.copy()
        translation_lmk[:, 0] = (
            lmk[:, 0] - areas[area_idx][0])*up_scale[area_idx]/down_scale
        translation_lmk[:, 1] = (
            lmk[:, 1] - areas[area_idx][1])*up_scale[area_idx]/down_scale
        return area_idx, translation_bbox.clip(min=0), translation_lmk.clip(min=0)

    def __non_max_suppression(self, boxes, overlap_thresh):
        if len(boxes) == 0:
            return []
        pick = []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = sorted(range(len(boxes)), reverse=True)
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]
            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlap_thresh)[0])))
        return pick
