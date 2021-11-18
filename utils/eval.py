import numpy as np

def floodFill(map, mask, curr_y, curr_x, cluster, id, map_obj, pixel_indices): # map gets masked distance map
  if curr_y < 0 or curr_y >= map.shape[0] or curr_x < 0 or curr_x >= map.shape[1]:
    return
  if mask[curr_y, curr_x] == False:
    return
  cluster.add((map[curr_y, curr_x], curr_y, curr_x))
  pixel_indices.add((curr_y, curr_x))
  map_obj[curr_y, curr_x] = id
  mask[curr_y, curr_x] = False
  for i in range(4):
    floodFill(map, mask, curr_y+dir[i][0], curr_x+dir[i][1], cluster, id, map_obj, pixel_indices)
  return
    

# Getting separate building objects
def getBuildings(distance_map): # applies in same way with gt and predicted
  '''
  Use floodfill() to distinguish building pixels
  '''
  mask = (distance_map >= 0)
  map_objects = np.zeros(distance_map.shape) # each building distinguished
  indices = set( [(distance_map[i[0], i[1]], i[0], i[1]) for i in np.argwhere(distance_map >= 0)]) # set of positive indices as a tuple
  clusters = {}
  k = 1 # building number
  while len(indices) != 0:
    tmp = set()
    pixel_indices = set()
    p = max(indices)
    floodFill(distance_map, mask, p[1], p[2], tmp, k, map_objects, pixel_indices)
    #get rid of indices in tmp from list indices
    clusters[k] = pixel_indices
    k += 1
    indices = indices-(indices & tmp)

  return clusters, map_objects


# get model evaluation metrics
def getMetrics(gt_label, pred_label):
  processed = dict.fromkeys([i for i in range(1,len(pred_label)+1)]) # key : pred_building id, val : gt_building id
  confusion = {'tp':0, 'fp':0, "tn":0, 'fn':0}
  iou = dict.fromkeys([i for i in range(1,len(gt_label)+1)]) # iou of each gt_ building label
  for j in range(1,len(pred_label)+1):
    working_set = set()
    for i in range(1, len(gt_label)+1):
      #calculate iou
      if len(gt_label[i] & pred_label[j]) != 0 and i not in set(processed.values()): # should have overlapping pixels and gt should not be processed
        intersection = gt_label[i] & pred_label[j]
        union = gt_label[i] | pred_label[j]
        curr_iou = len(intersection) / len(union)
        working_set.add((curr_iou, j, i))
    if len(working_set) == 0:
      confusion['fp'] += len(pred_label[j])
    else:
      s = max(working_set)
      if s[0] >= 0.5: #  true positive
        confusion['tp'] += len(pred_label[s[1]])
        iou[working_set[2]] = s[0]
        processed[working_set[1]] = s[2]
      else:
        confusion['fp'] += len(pred_label[s[1]]) # predicted building location is wrong
        processed[s[1]] = -1
        iou[s[2]] = s[0]

  gt_processed = set(processed.values()) # get correctly labeled gt buildings
  for gt in range(1,len(gt_label)+1):
    if gt not in gt_processed: # false negative
      confusion['fn'] += len(gt_label[gt])

  confusion['tn'] = 162*162 - confusion['tp'] - confusion['fp'] - confusion['fn']
  return confusion, iou, processed # confusion matrix, iou of gt_buildings