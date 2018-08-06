import numpy as np
from sklearn.metrics import confusion_matrix 
sptial_video_level_pred = np.load("meitu_rgb.npz")
rgb = sptial_video_level_pred['scores']
flow = sptial_video_level_pred['scores']
video_level_labels = sptial_video_level_pred['labels']
video_level_preds = [np.argmax(np.mean(x[0],axis=0)+np.mean(flow[i][0],axis=0)) for i,x in enumerate(rgb)]
cf = confusion_matrix(video_level_labels, video_level_preds).astype(float)
cls_cnt = cf.sum(axis=1)
cls_hit = np.diag(cf)
cls_acc = cls_hit/cls_cnt
print(cls_acc)
print('Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))
