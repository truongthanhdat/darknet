from YOLO import YOLO
from skimage import io

yolo = YOLO('data/yolo.cfg', 'data/yolo.weights', 'data/coco.data')
img = io.imread('dog.jpg')
res = yolo.detect(img)

print res
