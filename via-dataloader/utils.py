import os
import json
from functools import reduce  # forward compatibility for Python 3
import operator
import pprint
import shutil
import skimage.io
import numpy as np

def compute_center_distance(box, boxes):
    y_c = (box[2] + box[0]) / 2
    x_c = (box[3] + box[1]) / 2
    ys_c = (boxes[:, 2] + boxes[:, 0]) / 2
    xs_c = (boxes[:, 3] + boxes[:, 1]) / 2
    distances = np.square(ys_c - y_c) + np.square(xs_c - x_c)
    return distances


def compute_distances(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    distance_mat = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(distance_mat.shape[1]):
        box2 = boxes2[i]
        distance_mat[:, i] = compute_center_distance(box2, boxes1)
    return distance_mat


def extract_bbox_from_polygon(points_x, points_y, h, w):
        # should exclude the boundary points
        points_x = np.array(points_x)
        points_y = np.array(points_y)
        x_max = np.minimum(np.amax(points_x) + 1, w)
        x_min = np.maximum(np.amin(points_x) - 1, 0)
        y_max = np.minimum(np.amax(points_y) + 1, h)
        y_min = np.maximum(np.amin(points_y) - 1, 0)
        return np.array([y_min, x_min, y_max, x_max])

def compute_keypoints_boxes_distances(boxes, keypoints):
    # boxes: [N1, 4]
    # keypoints: [N2, 4, (x, y, v)]
    # return [N1, N2]
    def compute_points_box_distance(keypoints, box):
        # check keypoints
        # box: (y1, x1, y2, x2).
        # convert points as [[[x1, y1], [x2, y2], ...]]
        points = keypoints[:,:,:2]
        points_x = np.mean(points[:, :, 0], axis=1)
        points_y = np.mean(points[:, :, 1], axis=1)
        y_c = (box[2] + box[0]) / 2
        x_c = (box[3] + box[1]) / 2
        distances = np.sqrt(np.square(points_y - y_c) + np.square(points_x - x_c))
        return distances

    distance_mat = np.zeros((boxes.shape[0], keypoints.shape[0]))
    for i in range(distance_mat.shape[0]):
        box = boxes[i]
        distance_mat[i, :] = compute_points_box_distance(keypoints, box)
    return distance_mat

def compute_points_inside_boxes(boxes, keypoints):
    # boxes: [N1, 4]
    # keypoints: [N2, 4, (x, y, v)]
    # return [N1, N2]
    def compute_points_inside(keypoints, box):
        # check keypoints
        # box: (y1, x1, y2, x2).
        # convert points as [[[x1, y1], [x2, y2], ...]]
        points = keypoints[:,:,:2]
        check = np.logical_and(points >= box[[1, 0]], points <= box[[3, 2]])
        if_inside = np.all(np.logical_and(points >= box[[1, 0]], points <= box[[3, 2]]), axis=(1,2))
        return if_inside

    inside_mat = np.zeros((boxes.shape[0], keypoints.shape[0]))
    for i in range(inside_mat.shape[0]):
        box = boxes[i]
        inside_mat[i, :] = compute_points_inside(keypoints, box)
    return inside_mat

class VIAConverter:
    def __init__(self, json_path, dataset_path=None):
        self.json_path = json_path
        self.dataset_path = dataset_path


    def load_json(self):
        """
        Load the json dictionary from the json file.
        :return: None
        """
        self.annotations = json.load(open(self.json_path))

    def combine_multi_anno_files(self, list_files):
        self.annotations = {}
        for json_file in list_files:
            self.annotations.update(json.load(open(json_file)))


    def remove_key_from_annotations(self, key_to_remove_list):
        """
        Remove a certain key from the annotations
        :param key_to_remove: A list of keys from the root of the dictionary
        An example: ['file_attributes', 'number'] will remove the number key
        :return:
        """

        def getFromDict(dataDict, mapList):
            return reduce(operator.getitem, mapList, dataDict)
        # in case we need to set the key value
        def setInDict(dataDict, mapList, value):
            getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value
        # loop over each annotation
        for k, v in self.annotations.items():
            # pop the key (the last element in the key list)
            getFromDict(v, key_to_remove_list[:-1]).pop(key_to_remove_list[-1], None)



    def sort_clockwise(self, points):
        def angle_with_start(coord, start):
            vec = coord - start
            return np.angle(np.complex(vec[0], vec[1]))
        # convert into a coordinate system
        # (1, 1, 1, 2) -> (1, 1), (1, 2)
        coords = points.tolist()
        coords = [np.array(coord) for coord in coords]
        # make sure the first coord is the left shoulder
        start = coords[0]
        # sort the remaining coordinates by angle
        # with reverse=True because we want to sort by clockwise angle
        rest = sorted(coords[1:], key=lambda coord: angle_with_start(coord, start), reverse=False)

        # our first coordinate should be our starting point
        rest.insert(0, start)

        points = np.stack(rest)
        # convert into the proper coordinate format
        # (1, 1), (1, 2) -> (1, 1, 1, 2)
        return points


    def match_boxes(self, persons, keypoints, digits, digits_bboxes):
        """
        Match the bounding boxes between digit, person and keypoints
        :return:
        """
        numbers = []
        distances = compute_distances(np.array(persons), np.array(digits_bboxes))
        # for each digit, get the person id
        ids_associated_by_center_distance = np.argmin(distances, axis=0)
        #        ids_associated_by_overlaps = np.argmax(overlaps, axis=0)
        associated_person = ids_associated_by_center_distance
        # shape: [num person boxes, num keypoint annotations]
        output_kpts = []  # same number of persons
        output_digits = []  # same number of persons
        output_digit_boxes = []  # same number of persons
        for i in range(len(persons)):
            numbers.append("")
            output_kpts.append(np.zeros((4, 3), dtype=np.int32).tolist())
            output_digits.append([])
            output_digit_boxes.append([])
        if len(keypoints) > 0:
            # shape: [N_boxes, N_kpt_map]
            kpts_distances = compute_keypoints_boxes_distances(np.array(persons), np.array(keypoints))
            kpts_inside_mask = compute_points_inside_boxes(np.array(persons), np.array(keypoints))
            kpts_distances = np.where(kpts_inside_mask, kpts_distances, float('inf'))
            # print(kpts_distances)
            sorted_kpts_ids = np.argsort(np.sum(kpts_distances != float('inf'), axis=0))
            # print(sorted_kpts_ids)
            # match one by one for each keypoint map
            person_matches = [False] * len(persons)
            matched_person_ids = []
            for id in sorted_kpts_ids:
                # sord distances for each
                # sorted_distances = np.sort(kpts_distances[:, id])
                sorted_person_ids = np.argsort(kpts_distances[:, id])
                # print(sorted_person_ids)
                # print(sorted_distances)
                for p_id in sorted_person_ids:
                    if not person_matches[p_id]:
                        matched_person_ids.append(p_id)
                        person_matches[p_id] = True
                        break
            matched_person_ids = np.array(matched_person_ids)
            assert matched_person_ids.shape == np.unique(
                matched_person_ids).shape, "Wrong keypoint match on image."
            for idx, person_id in enumerate(matched_person_ids):
                output_kpts[person_id] = keypoints[sorted_kpts_ids[idx]]
        # generate numbers from associations, for each person roi (even no association)
        for idx, person_id in enumerate(associated_person):
            numbers[person_id] = numbers[person_id] + digits[idx]
            output_digits[person_id].append(digits[idx])
            output_digit_boxes[person_id].append(digits_bboxes[idx])

        return numbers, output_kpts, output_digits, output_digit_boxes

    def process_regions(self, regions_anno, height, width, filename=None):
        persons = []
        keypoints = []
        polygons = []
        digits = []
        numbers = []
        digits_bboxes = []

        for region in regions_anno:
            # first check the label type
            label = region['region_attributes']['label']
            if label == 'digit':
                # class label
                digits.append(region['region_attributes']['digit'])
                # digit bounding box
                try:  # original mask annotation
                    polygons.append(region["shape_attributes"])
                    digit_bbox = extract_bbox_from_polygon(region["shape_attributes"]["all_points_x"], \
                                                           region["shape_attributes"]["all_points_y"], \
                                                           height, width).tolist()
                except:  # bbox annotation
                    x1, x2, y1, y2 = region["shape_attributes"]["x"], \
                                     region["shape_attributes"]["x"] + region["shape_attributes"][
                                         "width"], \
                                     region["shape_attributes"]["y"], \
                                     region["shape_attributes"]["y"] + region["shape_attributes"][
                                         "height"]
                    digit_bbox = [y1, x1, y2, x2]
                digits_bboxes.append(digit_bbox)
            elif label == 'person':
                x1, x2, y1, y2 = region["shape_attributes"]["x"], \
                                 region["shape_attributes"]["x"] + region["shape_attributes"][
                                     "width"], \
                                 region["shape_attributes"]["y"], \
                                 region["shape_attributes"]["y"] + region["shape_attributes"][
                                     "height"]
                persons.append([y1, x1, y2, x2])
            else: # label is keypoint
                # print(region)
                p = region["shape_attributes"]
                # shape: (4, 3)
                kpts = np.stack((p['all_points_x'], p['all_points_y']), axis=-1)
                kpts = self.sort_clockwise(kpts)
                assert kpts.shape == (4, 2), "Wrong shape of keypoints on image {}.".format(filename)
                kpts = np.concatenate((kpts, np.ones((kpts.shape[0], 1), dtype=np.int8) * 2), axis=1)
                keypoints.append(kpts.tolist())

        # loop end
        numbers, output_kpts, output_digits, output_digit_boxes = self.match_boxes(persons, keypoints, digits,
                                                                                   digits_bboxes)
        return persons, polygons, numbers, output_kpts, output_digits, output_digit_boxes
        # numbers, output_kpts, output_digits, output_digit_boxes = self.match_boxes(persons, keypoints, digits, digits_bboxes)
        # output_anno = {'filename': a['filename'], \
        #                               'width': width, 'height': height, 'polygons': polygons, \
        #                               'keypoints': output_kpts, 'persons': persons, 'digits': digits, 'associated_person': associated_person.tolist(), 'numbers': numbers,
        #                               'digits_bboxes': digits_bboxes, 'video_id': a['file_attributes']['video_id']}
        # self.output_annotations.append()

    def convert_via_annotations(self):
        """
        Convert the via annotations to a better format, with verifications of annotations.
        :return:
        """
        output_annotations = []
        for _, anno in self.annotations.items():
            filename = anno['filename']
            print(filename)
            # only process annotation with regions label
            if anno['regions']:
                image_path = os.path.join(self.dataset_path, anno['filename'])
                image = skimage.io.imread(image_path)
                height, width = image.shape[:2]
                persons, polygons, numbers, output_kpts, output_digits, output_digit_boxes = self.process_regions(anno['regions'], height, width, filename=filename)
                res = {'filename': filename, 'width': width, 'height': height, 'polygons': polygons, \
                       'keypoints': output_kpts, 'persons': persons, 'digits': output_digits,
                       'digits_bboxes': output_digit_boxes, 'numbers': numbers, 'video_id': anno['file_attributes']['video_id']
                       }
                output_annotations.append(res)
                # copy image to the SJNM folder
                self.copy_image_to_path(image_path, r'D:\research\SJNM')
        return output_annotations

    def copy_image_to_path(self, image_old_path, image_folder):
        """
        We have different image folders for different batches, one easy way is to move all the images into one folder.
        :param image_old_path:
        :param image_folder:
        :return:
        """
        if not os.path.exists(image_folder):
            raise Exception("No such folder")
        # create new image path
        basename = os.path.basename(image_old_path)
        image_new_path = os.path.join(image_folder, basename)
        # for those images are not in the target folder, do the copy
        if not os.path.exists(image_new_path):
            shutil.copyfile(image_old_path, image_new_path)


    def via_old2new(self, json_file):
        """
        Convert old VIA project json to the new format with label attribute
        :param json_file:
        :return:
        """
        old_annotations = json.load(open(json_file))
        import copy
        new_annotations = copy.deepcopy(old_annotations)
        for k, v in old_annotations['_via_img_metadata'].items():
            # remove file attributes
            for target_key in ['Number', 'number', 'single']:
                new_annotations['_via_img_metadata'][k]['file_attributes'].pop(target_key, None)
            if v['regions']:
                for i, region in enumerate(v['regions']):
                    for key, val in region['region_attributes'].items():
                        region_attrs = new_annotations['_via_img_metadata'][k]['regions'][i]['region_attributes']
                        # check the entry properties
                        if (key == "digit" or key == "digits") and val != None:
                            region_attrs.pop("keypoints", None)
                            region_attrs.pop("person", None)
                            region_attrs['digit'] = region_attrs.pop("digits", None)
                            region_attrs['label'] = 'digit'
                        elif key == "keypoints" and val == "true":
                            region_attrs['label'] = 'keypoints'
                            region_attrs.pop("keypoints", None)
                            region_attrs.pop("person", None)
                        elif key == "person" and val == "true":
                            region_attrs['label'] = 'person'
                            region_attrs.pop("keypoints", None)
                            region_attrs.pop("person", None)
                        else:
                            # actually there are several files with wrong annotation type, did it manually
                            pass
                            # raise Exception("Annotation format incorrect on image {}".format(v['filename']))
        return new_annotations



    def test_print(self):
        pp = pprint.PrettyPrinter(indent=2)
        # pp.pprint(self.annotations["nba01_35_0.png257617"])
        pp.pprint(self.annotations["nba01_35_0.png257617"])

def save(ds_to_dump, save_dir="./", file_name='processed_via_total.json'):
    # basename = os.path.basename(self.json_path) # with .json ext
    with open(os.path.join(save_dir, "{}".format(file_name)), "w") as write_file:
        json.dump(ds_to_dump, write_file)

def convert_single_via_project():
    json_path = r"D:\research\playground-mask-rcnn\json\batch5.json"
    dataset_path = r"D:\research\batch_nba_01"
    via_converter = VIAConverter(json_path, dataset_path)
    via_converter.load_json()
    via_converter.remove_key_from_annotations(['file_attributes', 'number'])
    via_converter.remove_key_from_annotations(['file_attributes', 'single'])
    output_annotations = via_converter.convert_via_annotations()
    save(output_annotations, file_name="processed_batch5.json")

def process_multi_batch_files(list_batch_files, list_data_paths):
    output_annotations = {}
    i = 0
    for batch_file, data_path in zip(list_batch_files, list_data_paths):
        cur_converter = VIAConverter(batch_file, data_path)
        cur_converter.load_json()
        cur_annotations = cur_converter.convert_via_annotations()
        for annotation in cur_annotations:
            output_annotations[i] = annotation
            i += 1
    save(output_annotations, file_name="batch_all.json")



if __name__ == '__main__':
    batch_files = [r'D:\research\playground-mask-rcnn\json\batch_all.json', r'D:\research\playground-mask-rcnn\json\batch5.json']
    data_paths = [r'D:\research\SJNM', r'D:\research\batch_nba_01']
    process_multi_batch_files(batch_files, data_paths)

    # # via_converter.test_print()
    # via_converter.convert_via_annotations()
    # via_converter.save()
    # new_annotations = via_converter.via_old2new(r"D:\research\playground-mask-rcnn\json\via_total.json")
    # via_converter.save(new_annotations)
