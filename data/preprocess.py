import libs.solaris as sol # modified solaris library
import skimage
import os
from glob import glob
from utils.dist import *

SNdir = '/content/gdrive/MyDrive/SpaceNet/'

# for train mask
preprocess_trainmask = os.path.join(SNdir, 'processed/train_label_mask')

# for train datas
preprocess_train = os.path.join(SNdir, 'processed/train')
preprocess_trainlabel = os.path.join(SNdir, 'processed/train_label') # dest


# for test datas
preprocess_test = os.path.join(SNdir, 'processed/test')
preprocess_testlabel = os.path.join(SNdir, 'processed/test_label') # dest

train_dir = os.path.join(SNdir, 'AOI_5_Khartoum_Train')
test_dir = os.path.join(SNdir, 'AOI_5_Khartoum_Test_public')

rasterDir = os.path.join(train_dir, 'MUL-PanSharpen')
vectorDir = os.path.join(train_dir, 'geojson/buildings')

vectorList = glob(os.path.join(vectorDir, '*.geojson'))
rasterList = glob(os.path.join(rasterDir, '*.tif'))
vectorList.sort()
rasterList.sort()


for idx in range(0, len(rasterList)):
  imgName = rasterList[idx].split('/')[-1].split('.')[0].split('_')[-1]
  outfile = os.path.join(preprocess_trainlabel, imgName)
  # create sigend dist transform label 
  create_dist_map(rasterList[idx], vectorList[idx], npDistFileName=outfile)
  mul_img = sol.utils.io.imread(rasterList[idx]) # type : numpy ndarray

  # get mask and resize
  mask = sol.vector.mask.footprint_mask(vectorList[idx], reference_im=rasterList[idx])
  mask = skimage.transform.resize(mask, (650, 650))

  distanceMapped =  np.load(os.path.join(preprocess_trainlabel, imgName+'.npy'))
  # 90 degree flip
  img_90 = np.rot90(mul_img, axes=(1,0))
  mask_90 = np.rot90(mask, axes=(1,0))
  dist_90 = np.rot90(distanceMapped, axes=(1,0))
  # 180 degree roate
  img_180 = np.rot90(img_90, axes=(1,0))
  mask_180 = np.rot90(mask_90, axes=(1,0))
  dist_180 = np.rot90(dist_90, axes=(1,0))
  # 270 degree rotate
  img_270 = np.rot90(img_180, axes=(1,0))
  mask_270 = np.rot90(mask_180, axes=(1,0))
  dist_270 = np.rot90(dist_180, axes=(1,0))

  # save to directories
  np.save(os.path.join(preprocess_train,imgName), mul_img)
  np.save(os.path.join(preprocess_train,imgName+'_90'), img_90)
  np.save(os.path.join(preprocess_train,imgName+'_180'), img_180)
  np.save(os.path.join(preprocess_train,imgName+'_270'), img_270)

  np.save(os.path.join(preprocess_trainmask,imgName), mask)
  np.save(os.path.join(preprocess_trainmask,imgName+'_90'), mask_90)
  np.save(os.path.join(preprocess_trainmask,imgName+'_180'), mask_180)
  np.save(os.path.join(preprocess_trainmask,imgName+'_270'), mask_270)

  np.save(os.path.join(preprocess_trainlabel,imgName+'_90'), dist_90)
  np.save(os.path.join(preprocess_trainlabel,imgName+'_180'), dist_180)
  np.save(os.path.join(preprocess_trainlabel,imgName+'_270'), dist_270)