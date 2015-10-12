import cfg as c
import numpy as np
import scipy.io as sio

from PIL import Image
from pdb import set_trace as T

from utils import FeatureCollection
from generate_matrices import rbs, get_xy


LOGO_FILE = 'data/cal-logo-bw.jpg'
X, Y = (20, 20) # Size of image
XS, YS = (1, 11)   # Location of upper left pixel on the image

def get_image_matrix():
    img = Image.open(LOGO_FILE)
    img.convert('L') # To grayscale
    # To array
    data = np.reshape(np.array(map(np.mean, img.getdata())), (X, Y))

    # Pad
    ypad = (YS, c.YSIZE - YS - Y)
    xpad = (XS, c.XSIZE - XS - X)
    data = np.pad(data, (xpad, ypad), mode='constant', constant_values=0)
    assert(data.shape == (c.XSIZE, c.YSIZE))

    # Normalize
    data = 1 - data/255.0
    return data


def test_image_matrix(station_names):
    data = get_image_matrix()
    fc = FeatureCollection()
    for n in station_names:
        x, y = get_xy(n)
        fc.add_polygon(rbs.get_poly(x, y), dict(weight=data[x][y]))
    fc.dump('tmp1.geojson')

def get_availabilities(station_names):
    data = get_image_matrix()
    def get_avail(name):
        x, y = get_xy(name)
        return data[x][y]
    return map(get_avail, station_names)

if __name__ == '__main__':
    test_image_matrix(sio.loadmat('queueing_params.mat')['stations'])
