from functools import partial
from osgeo import osr, ogr, gdalnumeric, gdal
from shapely.geometry.polygon import Polygon
import shapely
import pyproj
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage

def utm_getZone(longitude):
    return (int(1+(longitude+180.0)/6.0))

def utm_isNorthern(latitude):
    if (latitude < 0.0):
        return 0
    else:
        return 1

def getUTMcrs(polyGeom, srcCrs="+proj=longlat +datum=WGS84 +no_defs"):

    wgs84CRS = "+proj=longlat +datum=WGS84 +no_defs" # EPGS 4326

    # need to be corrected to right coordinate. Set to EPGS 4326
    projectTO_WGSBase = pyproj.Transformer.from_crs(srcCrs, wgs84CRS, always_xy=True).transform
    polyGeomWGS = shapely.ops.transform(projectTO_WGSBase, polyGeom)


    polyCentroid = polyGeomWGS.centroid
    utm_zone = utm_getZone(polyCentroid.x)
    is_northern = utm_isNorthern(polyCentroid.y)
    if is_northern:
        directionIndicator = '+north'
    else:
        directionIndicator = '+south'

    utm_crs = "+proj=utm +zone={} {} +ellps=WGS84 +datum=WGS84 +units=m +no_defs".format(utm_zone,
                                                                                        directionIndicator)  
    return utm_crs

  
def create_dist_map(rasterSrc, vectorSrc, npDistFileName='', 
                           noDataValue=0, burn_values=1, 
                           dist_mult=1, vmax_dist=64):

    '''
    Create building signed distance transform from Yuan 2016 
    (https://arxiv.org/pdf/1602.06564v1.pdf).
    vmax_dist: absolute value of maximum distance (meters) from building edge
    Adapted from createNPPixArray in labeltools
    '''

    ## open source vector file that truth data
    source_ds = ogr.Open(vectorSrc)
    source_layer = source_ds.GetLayer()

    ## extract data from src Raster File to be emulated
    ## open raster file that is to be emulated
    srcRas_ds = gdal.Open(rasterSrc)
    upx, xres, xskew, upy, yskew, yres = srcRas_ds.GetGeoTransform()
    cols = srcRas_ds.RasterXSize
    rows = srcRas_ds.RasterYSize
    
    ulx = upx +0*xres + 0*xskew
    uly = upy + 0*yskew + 0*yres
    lrx = upx + cols*xres + rows*xskew  
    lry = upy + cols*yskew + rows*yres

    poly = Polygon([(ulx, uly),
                   (lrx, uly),
                   (lrx, lry),
                   (ulx, lry),
                    (ulx, uly)]
                    )

    geoTrans = srcRas_ds.GetGeoTransform()
    
    # Get utm_crs
    utm_crs = getUTMcrs(poly)                       
    # Get meterindex for use in distance transformation below
    line = ogr.Geometry(ogr.wkbLineString)
    line.AddPoint(geoTrans[0], geoTrans[3])
    line.AddPoint(geoTrans[0]+geoTrans[1], geoTrans[3])

    # create transform
    start = osr.SpatialReference()
    start.ImportFromEPSG(4326)
    target = osr.SpatialReference()
    target.ImportFromProj4(utm_crs)
    tranform_WGS84_To_UTM = osr.CoordinateTransformation(start, target)
    line.Transform(tranform_WGS84_To_UTM)
    metersIndex = line.Length()

    memdrv = gdal.GetDriverByName('MEM') # temporary in-memory format

    # Temp result raster
    dst_ds = memdrv.Create('', cols, rows, 1, gdal.GDT_Byte)
    dst_ds.SetGeoTransform(srcRas_ds.GetGeoTransform())
    dst_ds.SetProjection(srcRas_ds.GetProjection())
    band = dst_ds.GetRasterBand(1)
    band.SetNoDataValue(noDataValue)

    gdal.RasterizeLayer(dst_ds, [1], source_layer, burn_values=[burn_values])
    srcBand = dst_ds.GetRasterBand(1)

    memdrv2 = gdal.GetDriverByName('MEM')
    prox_ds = memdrv2.Create('', cols, rows, 1, gdal.GDT_Int16)
    prox_ds.SetGeoTransform(srcRas_ds.GetGeoTransform())
    prox_ds.SetProjection(srcRas_ds.GetProjection())
    proxBand = prox_ds.GetRasterBand(1)
    proxBand.SetNoDataValue(noDataValue)

    opt_string = 'NODATA='+str(noDataValue)
    options = [opt_string]

    gdal.ComputeProximity(srcBand, proxBand, options)

    memdrv3 = gdal.GetDriverByName('MEM')
    proxIn_ds = memdrv3.Create('', cols, rows, 1, gdal.GDT_Int16)
    proxIn_ds.SetGeoTransform(srcRas_ds.GetGeoTransform())
    proxIn_ds.SetProjection(srcRas_ds.GetProjection())
    proxInBand = proxIn_ds.GetRasterBand(1)
    proxInBand.SetNoDataValue(noDataValue)
    opt_string2 = 'VALUES='+str(noDataValue)
    options = [opt_string, opt_string2]

    gdal.ComputeProximity(srcBand, proxInBand, options)

    proxIn = gdalnumeric.BandReadAsArray(proxInBand)*metersIndex
    proxOut = gdalnumeric.BandReadAsArray(proxBand)*metersIndex # current pixel distance from nearest buildings

    # rescale pixel values btw -1~+1, check for nan

    mask_array = gdalnumeric.BandReadAsArray(srcBand)
    proxIn = proxIn / np.max(proxIn)
    proxOut = proxOut / np.max(proxOut)
    if len(np.argwhere(mask_array != 0)) < 2:
        proxIn = np.zeros((cols,rows))
        proxOut = np.ones((cols,rows)) * 1.        
    proxTotal = proxIn.astype(np.float32) - proxOut.astype(np.float32)
    # save
    if npDistFileName != '':
        # save as npy file since some values will be negative
        np.save(npDistFileName, proxTotal)
    return


def buildingMasked(input_img, mask, signed_distance=False, alpha=0.5): # larger alpha emphasizes  input_img
    # use mask input as sol.vector.mask.footprint_mask
    building_mask = (mask != 0.) # mark pixels as True/False (building)
    color = 10
    if not signed_distance:
        color = [255, 0, 0]
        alpha = 0.3
    overlay = input_img.copy()
    overlay[building_mask] = np.array(color, dtype=np.uint8)
    result = input_img.copy()
    cv2.addWeighted(overlay, alpha, result, 1-alpha, 0, result)
    return result 