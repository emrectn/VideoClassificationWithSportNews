"""
Classify a few images through our CNN.
"""
import numpy as np
import operator
import random
import glob
import os.path
from data import DataSet
from processor import process_image
from keras.models import load_model
import cv2
import imageio
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

def putText(frame,a,b):
    height, width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    color = (255, 255, 255)
    cv2.putText(frame, (a), (10,20), font, 1, color, 2, cv2.LINE_AA)
    cv2.putText(frame, ("CCC MURAT REIS"), (0, height-10), font, 1, color, 2, cv2.LINE_AA)
    return frame

def detect(image,model,data):
    # Turn the image into an array.
    image_arr = process_image(image, (299, 299, 3))
    image_arr = np.expand_dims(image_arr, axis=0)
    # Predict.
    predictions = model.predict(image_arr)
    # Show how much we think it's each one.
    label_predictions = {}
    for i, label in enumerate(data.classes):
        print(i,label)
        label_predictions[label] = predictions[0][i]

    sorted_lps = sorted(label_predictions.items(), key=operator.itemgetter(1), reverse=True)

    for i, class_prediction in enumerate(sorted_lps):
        # Just get the top five.
        if i > 0:
            break
        a = class_prediction[0]
        b = class_prediction[1]
        i += 1
    return a,b

def main(nb_images=5):
    """Spot-check `nb_images` images."""
    data = DataSet()
    model = load_model('data/checkpoints/inception.001-0.24.hdf5')
    video_name = 'v_Deneme_g01_c01'
    video_class = "Deneme"
    video = cv2.VideoCapture("data/test/" + video_class + "/" + video_name + ".avi")
    fps = video.get(cv2.CAP_PROP_FPS)
    writer = imageio.get_writer(video_name + "_predict.avi",fps = fps)
    images = glob.glob(os.path.join('data', 'test', video_class,video_name + '-*.jpg'))
    print(images)
    for i,x in enumerate(images):
        a,b = detect(x,model,data)
        print(i, x, a, b)
        frame = cv2.imread(x)
        frame = putText(frame,a,b)
        scene = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(scene)


if __name__ == '__main__':
    main()
