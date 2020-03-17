from .yolo import Detector


def get_detector(detector_type='yolo'):
    if detector_type == 'yolo':
        return Detector()
