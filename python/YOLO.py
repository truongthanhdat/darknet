import darknet
import numpy as np
import ctypes

class YOLO:
    def __init__(self, netPath, weightsPath, metaPath):
        self.net = darknet.load_net(netPath, weightsPath, 0)
        self.meta = darknet.load_meta(metaPath)
        self.boxes = darknet.make_boxes(self.net)
        self.probs = darknet.make_probs(self.net)
        self.num = darknet.num_boxes(self.net)

    def __del__(self):
        print 'Release YOLO Object'
        darknet.free_ptrs(ctypes.cast(self.probs, darknet.POINTER(ctypes.c_void_p)), self.num)

    def detect(self, img, thresh = .5, hier_thresh = .5, nms = .45):
        img = img.astype(np.float32)
        im = darknet.convertImage(img.ctypes.data, img.shape[0], img.shape[1], img.shape[2])
        darknet.network_detect(self.net, im, thresh, hier_thresh, nms, self.boxes, self.probs)
        res = []
        for j in xrange(self.num):
            for i in xrange(self.meta.classes):
                if self.probs[j][i] > 0:
                    res.append((self.meta.names[i], self.probs[j][i], (self.boxes[j].x, self.boxes[j].y, self.boxes[j].w, self.boxes[j].h)))

        darknet.free_image(im)
        return res
