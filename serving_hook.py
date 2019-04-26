import base64
import json
import logging

import cv2
import numpy as np


LOG = logging.getLogger(__name__)
PARAMS = {
    'device': 'CPU',
    'threshold': 0.7,
    'output_type': 'bytes',
    'need_table': True,
    'target_size': (544, 320)
}


def init_hook(**params):
    global PARAMS
    PARAMS.update(params)

    PARAMS['threshold'] = float(PARAMS['threshold'])

    PARAMS['need_table'] = _boolean_string(PARAMS['need_table'])
    LOG.info('Init with params:')
    LOG.info(json.dumps(PARAMS, indent=2))


def _boolean_string(s):
    if isinstance(s, bool):
        return s

    s = s.lower()
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


def _load_image(inputs, image_key):
    image = inputs.get(image_key)
    if image is None:
        raise RuntimeError('Missing "{0}" key in inputs. Provide an image in "{0}" key'.format(image_key))

    if len(image.shape) == 0:
        image = np.stack([image.tolist()])

    if len(image.shape) < 3:
        image = cv2.imdecode(np.frombuffer(image[0], np.uint8), cv2.IMREAD_COLOR)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def process(inputs, ctx, **kwargs):
    frame = _load_image(inputs, 'input')
    # convert to BGR
    data = frame[:, :, ::-1]
    data = cv2.resize(data, PARAMS['target_size'], interpolation=cv2.INTER_AREA)

    # convert to input shape (N, C, H, W)
    data = np.expand_dims(np.transpose(data, [2, 0, 1]), axis=0)

    input_name = list(kwargs['model_inputs'])[0]
    outputs = ctx.driver.predict({input_name: data})

    outputs = list(outputs.values())[0].reshape([-1, 7])
    # 7 values:
    # class_id, label, confidence, x_min, y_min, x_max, y_max
    # Select boxes where confidence > factor
    outputs = outputs.reshape(-1, 7)
    bboxes_raw = outputs[outputs[:, 2] > PARAMS['threshold']]
    bounding_boxes = bboxes_raw[:, 3:7]
    bounding_boxes[:, 0] = bounding_boxes[:, 0] * frame.shape[1]
    bounding_boxes[:, 2] = bounding_boxes[:, 2] * frame.shape[1]
    bounding_boxes[:, 1] = bounding_boxes[:, 1] * frame.shape[0]
    bounding_boxes[:, 3] = bounding_boxes[:, 3] * frame.shape[0]
    bounding_boxes = bounding_boxes.astype(int)

    result = {'person_boxes': bounding_boxes, 'person_scores': bboxes_raw[:, 2]}
    if len(bounding_boxes) > 0:
        table = result_table_string(result, frame)
        add_overlays(frame, bounding_boxes, labels=None)
    else:
        table = []

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if PARAMS['output_type'] == 'bytes':
        image_bytes = cv2.imencode(".jpg", frame, params=[cv2.IMWRITE_JPEG_QUALITY, 95])[1].tostring()
    else:
        image_bytes = frame

    return {
        'output': image_bytes,
        'table_output': table
    }


def add_overlays(frame, boxes, labels=None):
    font = cv2.FONT_HERSHEY_SIMPLEX

    if boxes is not None:
        for i, face in enumerate(boxes):
            face_bb = face.astype(int)
            cv2.rectangle(
                frame,
                (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                (0, 250, 0), thickness=2
            )

            if labels:
                cv2.putText(
                    frame,
                    labels[i],
                    (face_bb[0] + 4, face_bb[1] + 5),
                    font, 1.0, (0, 250, 0), thickness=1, lineType=cv2.LINE_AA,
                )


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    (h, w) = image.shape[0], image.shape[1]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def result_table_string(result_dict, frame):
    table = []
    w, h = frame.shape[1], frame.shape[0]

    def crop_from_box(box, normalized_coordinates=False):
        left, right = max(0, box[0]), min(box[2], w)
        top, bottom = max(0, box[1]), min(box[3], h)
        if normalized_coordinates:
            left, right = left * w, right * w
            top, bottom = top * h, bottom * h

        cropped = frame[top:bottom, left:right]
        # max size width 128
        if cropped.shape[1] > 128:
            cropped = image_resize(cropped, height=128)
        cim = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
        image_bytes = cv2.imencode(".jpg", cim, params=[cv2.IMWRITE_JPEG_QUALITY, 95])[1].tostring()

        return image_bytes

    def append(type_, name, prob, image):
        encoded = image
        if image is not None:
            encoded = base64.encodebytes(image).decode()

        table.append(
            {
                'type': type_,
                'name': name,
                'prob': float(prob),
                'image': encoded
            }
        )

    if len(result_dict.get('person_boxes', [])) > 0:
        for prob, box in zip(result_dict['person_scores'], result_dict['person_boxes']):
            append('person', 'person', prob, crop_from_box(box))

    return json.dumps(table)
