import cv2
from imwatermark import WatermarkEncoder

bgr = cv2.imread('../../assets/original.png')
wm = 'test'

encoder = WatermarkEncoder()
encoder.set_watermark('bytes', wm.encode('utf-8'))
bgr_encoded = encoder.encode(bgr, 'dwtDct')

cv2.imwrite('../../assets/test_wm.png', bgr_encoded)
