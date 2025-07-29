import cv2
import numpy as np


'''
img = cv2.imread("./test_img.jpg")  
retval, buf = cv2.imencode(".jpg", img)
buf = np.array(np.frombuffer(buf, dtype=np.uint8))
img = cv2.imdecode(buf, cv2.IMREAD_COLOR)

cv2.imshow("decoded image", img)
'''

import cv2
import numpy as np  

f = open('test_img.jpg', 'rb')
image_bytes = f.read()  # b'\xff\xd8\xff\xe0\x00\x10...'

# decoded = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)

# print('OpenCV:\n', decoded)

# your Pillow code
import io
from PIL import Image
image = np.array(Image.open(io.BytesIO(image_bytes))) 
print('PIL:\n', image)