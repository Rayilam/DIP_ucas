import os
os.system("python ./slim_face.py 1")
os.system("python ./big_eye.py 1")
os.system("python ./lighten.py 1")
if os.path.exists("beautify/slim_face.jpg"):
    os.remove("beautify/slim_face.jpg")
if os.path.exists("beautify/face_eye.jpg"):
    os.remove("beautify/face_eye.jpg")
if os.path.exists("beautify/9_eye.jpg"):
    os.remove("beautify/9_eye.jpg")

