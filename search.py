from extractor import online
import numpy as np
from initialize_engine import FAISS


def search(image):
    assert isinstance(image, np.ndarray)
    feature = online(image)
    faiss_instance = FAISS.instance()
    results = faiss_instance.search(feature)

    return results
    