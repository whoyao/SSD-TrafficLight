import tensorflow as tf
from nets import ssd_vgg_512
from nets import ssd_common
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as mpcm
import cv2
import numpy as np
import os

from preprocessing import ssd_vgg_preprocessing as ssd_preprocessing
from preprocessing import tf_image

slim = tf.contrib.slim
isess = tf.InteractiveSession()

ckpt_filename = '/home/luyifan/Project/Udacity/traffic_light_bag/logs/model.ckpt'

colors_tableau = [(255, 255, 255), (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]


def colors_subselect(colors, num_classes=21):
    dt = len(colors) // num_classes
    sub_colors = []
    for i in range(num_classes):
        color = colors[i * dt]
        if isinstance(color[0], float):
            sub_colors.append([int(c * 255) for c in color])
        else:
            sub_colors.append([c for c in color])
    return sub_colors


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """Draw a collection of lines on an image.
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def draw_rectangle(img, p1, p2, color=[255, 0, 0], thickness=2):
    cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)


def draw_bbox(img, bbox, shape, label, color=[255, 0, 0], thickness=2):
    p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
    p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
    cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
    p1 = (p1[0] + 15, p1[1])
    cv2.putText(img, str(label), p1[::-1], cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)


def bboxes_draw_on_img(img, classes, scores, bboxes, colors, thickness=2):
    shape = img.shape
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        color = colors[classes[i]]
        # Draw bounding box...
        p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
        p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
        cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
        # Draw text...
        s = '%s/%.3f' % (classes[i], scores[i])
        p1 = (p1[0] - 5, p1[1])
        cv2.putText(img, s, p1[::-1], cv2.FONT_HERSHEY_DUPLEX, 0.4, color, 1)


reuse = True if 'ssd' in locals() else None
# Input placeholder.
net_shape = (512, 512)
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))

image_pre, labels_pre, bboxes_pre, bbox_img = ssd_preprocessing.preprocess_for_eval(
    img_input, None , None, net_shape, resize=ssd_preprocessing.Resize.PAD_AND_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

params = ssd_vgg_512.SSDNet.default_params
ssd = ssd_vgg_512.SSDNet(params)

layers_anchors = ssd.anchors(net_shape, dtype=np.float32)

# Re-define the model
with slim.arg_scope(ssd.arg_scope(weight_decay=0.0005)):
    predictions, localisations, logits, end_points = ssd.net(image_4d, is_training=False, reuse=reuse)
    localisations = ssd.bboxes_decode(localisations, layers_anchors)
    scores, bboxes = \
            ssd.detected_bboxes(predictions, localisations,
                                select_threshold=0.5,
                                nms_threshold=0.35,
                                clipping_bbox=None,
                                top_k=400,
                                keep_top_k=200)




# Main processing routine.
def process_image(img, select_threshold=0.5, nms_threshold=0.35, net_shape=(512, 512)):
    # Run SSD network.
    rscores, rbboxes = isess.run([scores, bboxes], feed_dict={img_input: img})

    # Resize bboxes to original image shape.
    rbboxes = tf_image.bboxes_resize(rbboxes, img)

    return rscores, rbboxes

# Test on demo images.
path = '../test/'
image_names = sorted(os.listdir(path))
img = mpimg.imread(path + image_names[0])
resized_img = cv2.resize(img, (512, 512))

saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

rscores, rbboxes =  process_image(resized_img)
print(rscores)
print('--------------------------')
print(rbboxes)

# Draw results.
# img_bboxes = np.copy(img)
# bboxes_draw_on_img(img_bboxes, rclasses, rscores, rbboxes, colors_tableau, thickness=2)

# mpimg.imsave('output.jpeg', img_bboxes)

# fig = plt.figure(figsize = (12, 12))
# plt.imshow(img_bboxes)
