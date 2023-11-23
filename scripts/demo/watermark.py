import os, sys
BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)))
ASSETS = BASE_PATH.split(os.sep)[:-2]
ASSETS=f'{os.sep}'.join(ASSETS)
ASSETS=os.path.join(ASSETS,'assets')

import cv2
from imwatermark import WatermarkEncoder

bgr = cv2.imread(os.path.join(ASSETS,'original.jpg'))
wm = 'test'

encoder = WatermarkEncoder()
encoder.set_watermark('bytes', wm.encode('utf-8'))
bgr_encoded = encoder.encode(bgr, 'dwtDct')

cv2.imwrite(os.path.join(ASSETS,'original_wm.png'), bgr_encoded)
