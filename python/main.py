import darknet
import numpy as np
import ctypes
import skvideo.io as skv
import argparse
import progressbar

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='Input video', required=True)
parser.add_argument('--output', type=str, help='Output text', required=True)
parser.add_argument('--net', type=str, help='Path to net config', required=True)
parser.add_argument('--weights', type=str, help='Path to weights', required=True)
parser.add_argument('--meta', type=str, help='Path to meta', required=True)
args = parser.parse_args()

net = darknet.load_net(args.net, args.weights, 0)
meta = darknet.load_meta(args.meta)
boxes = darknet.make_boxes(net)
probs = darknet.make_probs(net)
num   = darknet.num_boxes(net)

def detect(img):
    img = img.astype(np.float32)
    im = darknet.convertImage(img.ctypes.data, img.shape[0], img.shape[1], img.shape[2])
    darknet.network_detect(net, im, 0.5, 0.5, 0.45, boxes, probs)
    res = []
    for j in xrange(num):
        for i in xrange(meta.classes):
            if probs[j][i] > 0:
                res.append(meta.names[i])
    darknet.free_image(im)
    return res

video = skv.vread(args.input)
bar = progressbar.ProgressBar()
index = bar(xrange(video.shape[0]))

with open(args.output, 'w') as output:
    for i in index:
        res = detect(video[i])
        if (len(res) > 0):
            string = ('frame_%d: ' % i) + ', '.join(res) + '\n'
            output.write(string)


darknet.free_ptrs(ctypes.cast(probs, darknet.POINTER(ctypes.c_void_p)), num)
