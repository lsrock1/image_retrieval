from fvcore.common.config import CfgNode as CN


_C = CN()

_C.FEATURE_PATH = 'resources/extracted_features'
_C.DB_IMAGES_PATH = 'resources/images'
_C.QUERY_IMAGES_PATH = 'resources/query_images'
_C.MODEL_PATH = 'resources/models/model.pth'

_C.MODEL = CN()
_C.MODEL.DEVICE = 'cuda'
_C.MODEL.NUM_CLASSES = 400
_C.MODEL.DIM = 2048
_C.MODEL.PCA = True
_C.MODEL.DB_FEATURE_SIZE = 10000
_C.MODEL.OFFLINE_BS = 100
