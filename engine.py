import faiss
from utils.singleton import SingletonInstance
import numpy as np
from pathlib import Path
import extractor
from utils.configs import cfg


class FAISS(SingletonInstance):
    def __init__(self, cfg):
        self.build_index(cfg)
        self.load_image_names(cfg)

    def build_index(self, cfg):
        self.index = faiss.IndexFlatL2(cfg.MODEL.DIM)
        feature_path = Path(cfg.FEATURE_PATH)
        path = list(feature_path.glob('*.npy'))
        for p in path:
            self.index.add(np.load(str(p)))
        self.__move_index_to_gpu(cfg)

    def __move_index_to_gpu(self, cfg):
        if cfg.MODEL.DEVICE != 'cpu':
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

    def load_image_names(self, cfg):
        feature_path = Path(cfg.FEATURE_PATH)
        path = str(feature_path / 'image_names.txt')
        self.image_names = []
        with open(path, 'r') as f:
            for r in f.readlines():
                r = r.strip()
                if len(r) > 0:
                    self.image_names.append(r)

    def search(self, feature, topk):
        assert isinstance(feature, np.ndarray)
        assert len(feature.shape) == 2
        D, I = index.search(feature, topk)
        
        # return index
        results = []
        for query in I:
            tmp = []
            for topn in query:
                tmp.append(self.load_image_names[topn])
            results.append(tmp)
        return results


def init_engine():
    cfg.merge_from_file('static_config.yaml')
    cfg.freeze()

    FAISS.instance(cfg)
    extractor.init_model(cfg)


def search(image):
    feature = extractor.online(image)
    faiss_instance = FAISS.instance()
    results = faiss_instance.search(feature)

    return results