from .model import extracting
from pathlib import Path


def offline(cfg):
    # update from all images
    # using mini batch
    db_image_path = Path(cfg.DB_IMAGES_PATH)
    db_images = db_image_path.glob('*')
    
    feature_results = []
    extracting()

    feature_path = Path(cfg.FEATURE_PATH)
    path = str(feature_path / 'image_names.txt')
    with open(path, 'w') as f:
        for image in db_images:
            f.write(str(image))


def online(cfg, image):
    feature = extracting(image)
    return feature.cpu().numpy()
