from nets import ssd_vgg_512
from nets import ssd_common

from preprocessing import ssd_vgg_preprocessing

ckpt_filename = '/home/luyifan/Project/Udacity/traffic_light_bag/logs/model.ckpt'

# SSD object.
reuse = True if 'ssd' in locals() else None
params = ssd_vgg_512.SSDNet.default_params
ssd = ssd_vgg_512.SSDNet(params)

# Image pre-processimg
out_shape = ssd.params.img_shape
image_pre, labels_pre, bboxes_pre, bbox_img = \
    ssd_vgg_preprocessing.preprocess_for_eval(image, labels, bboxes, out_shape,
                                              resize=ssd_vgg_preprocessing.Resize.CENTRAL_CROP)
image_4d = tf.expand_dims(image_pre, 0)

# SSD construction.
with slim.arg_scope(ssd.arg_scope(weight_decay=0.0005)):
    predictions, localisations, logits, end_points = ssd.net(image_4d, is_training=False, reuse=reuse)

# SSD default anchor boxes.
img_shape = out_shape
layers_anchors = ssd.anchors(img_shape, dtype=np.float32)

for k in sorted(end_points.keys()):
    print(k, end_points[k].get_shape())

# Targets encoding.
target_labels, target_localizations, target_scores = \
    ssd_common.tf_ssd_bboxes_encode(labels, bboxes_pre, layers_anchors,
                                    num_classes=params.num_classes, no_annotation_label=params.no_annotation_label)

nms_threshold = 0.5

# Output decoding.
localisations = ssd.bboxes_decode(localisations, layers_anchors)
tclasses, tscores, tbboxes = ssd_common.tf_ssd_bboxes_select(predictions, localisations)
tclasses, tscores, tbboxes = ssd_common.tf_bboxes_sort(tclasses, tscores, tbboxes, top_k=400)
tclasses, tscores, tbboxes = ssd_common.tf_bboxes_nms_batch(tclasses, tscores, tbboxes,
                                                            nms_threshold=0.3, num_classes=ssd.params.num_classes)

# Initialize variables.
init_op = tf.global_variables_initializer()
isess.run(init_op)
# Restore SSD model.
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# Run model.
[rimg, rpredictions, rlocalisations, rclasses, rscores, rbboxes, glabels, gbboxes, rbbox_img, rt_labels, rt_localizations, rt_scores] = \
    isess.run([image_4d, predictions, localisations, tclasses, tscores, tbboxes,
               labels, bboxes_pre, bbox_img,
               target_labels, target_localizations, target_scores])

def bboxes_select(classes, scores, bboxes, threshold=0.1):
    """Sort bounding boxes by decreasing order and keep only the top_k
    """
    mask = scores > threshold
    classes = classes[mask]
    scores = scores[mask]
    bboxes = bboxes[mask]
    return classes, scores, bboxes

print(rclasses, rscores)
print(rscores.shape)

rclasses, rscores, rbboxes = bboxes_select(rclasses, rscores, rbboxes, 0.1)
# print(list(zip(rclasses, rscores)))
# print(rbboxes)

# # Compute classes and bboxes from the net outputs.
# rclasses, rscores, rbboxes,_,_ = ssd_common.ssd_bboxes_select(rpredictions, rlocalisations, layers_anchors,
#                                                                threshold=0.3, img_shape=img_shape,
#                                                                num_classes=21, decode=True)
# rbboxes = ssd_common.bboxes_clip(rbbox_img, rbboxes)
# rclasses, rscores, rbboxes = ssd_common.bboxes_sort(rclasses, rscores, rbboxes, top_k=400, priority_inside=False)
# rclasses, rscores, rbboxes = ssd_common.bboxes_nms(rclasses, rscores, rbboxes, threshold=0.3)

# Draw bboxes
img_bboxes = np.copy(ssd_vgg_preprocessing.np_image_unwhitened(rimg[0]))
bboxes_draw_on_img(img_bboxes, rclasses, rscores, rbboxes, colors_tableau, thickness=1)
# bboxes_draw_on_img(img_bboxes, glabels, np.zeros_like(glabels), gbboxes, colors_tableau, thickness=1)
# bboxes_draw_on_img(img_bboxes, test_labels, test_scores, test_bboxes, colors_tableau, thickness=1)

print('Labels / scores:', list(zip(rclasses, rscores)))
print('Grountruth labels:', list(glabels))
print(gbboxes)
print(rbboxes)

fig = plt.figure(figsize = (10,10))
plt.imshow(img_bboxes)

import tf_extended as tfe

isess.run(ssd_common.tf_bboxes_jaccard(gbboxes[0], rbboxes))

test_bboxes = []
test_labels = []
test_scores = []
for i in range(0, 3):
    yref, xref, href, wref = layers_anchors[i]
    ymin = yref - href / 2.
    xmin = xref - wref / 2.
    ymax = yref + href / 2.
    xmax = xref + wref / 2.
    bb = np.stack([ymin, xmin, ymax, xmax], axis=-1)

    idx = yref.shape[0] // 2
    idx = np.random.randint(yref.shape[0])
    #     print(bb[idx, idx].shape)
    test_bboxes.append(bb[idx, idx])
    test_labels.append(np.ones(href.shape, dtype=np.int64) * i)
    test_scores.append(np.ones(href.shape))

test_bboxes = np.concatenate(test_bboxes)
test_labels = np.concatenate(test_labels)
test_scores = np.concatenate(test_scores)

print(test_bboxes.shape)
print(test_labels.shape)
print(test_scores.shape)

rt_labels, rt_localizations, rt_scores
for i in range(len(rt_labels)):
    print(rt_labels[i].shape)
    idxes = np.where(rt_labels[i] > 0)
#     idxes = np.where(rt_scores[i] > 0.)
    print(idxes)
    print(rt_localizations[i][idxes])
    print(list(zip(rt_labels[i][idxes], rt_scores[i][idxes])))

# fig = plt.figure(figsize = (8,8))
# plt.imshow(ssd_preprocessing.np_image_unwhitened(rimg[0]))
# print('Ground truth labels: ', rlabels)

# Request threads to stop. Just to avoid error messages
# coord.request_stop()
# coord.join(threads)