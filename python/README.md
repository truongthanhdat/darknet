# Tutorial

```python
import YOLO
```

First, you need to create new yolo detector.

```python
yolo = YOLO.YOLO('data/yolo.cfg', 'data/yolo.weights', 'data/coco.data')
#Arguments: Network Path, Weights Path, Meta Path
```

Second, you can detect object from image by YOLO.detect.

```python
from skimage import io
img = io.imread('dogs.jpg')
res = yolo.detect(img)
#Input: image with Blue - Green - Red channel
#Output: List of tubles: (name object, prob, (x, y, w, h))
#Note: If you read image or video by open CV, it returns BGR images. We need RGB image as input.
```

