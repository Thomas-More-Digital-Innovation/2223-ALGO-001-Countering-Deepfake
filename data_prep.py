import cv2
from os import walk, remove, listdir, path
import streamlit as st
import face_recognition
import numpy as np
from keras import utils
from keras.models import load_model
from sklearn.metrics import f1_score

INPUT_SIZE = (299, 299)
model = load_model('./deepfake_model.h5', custom_objects={"f1_score": f1_score})


# pulls images from the video
def generate_images_from_videos(video, extraction_path):
    try:
        cap = cv2.VideoCapture(video)
        current_frame = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        while (cap.isOpened()):
            success, frame = cap.read()
            cv2.imwrite(f"{extraction_path}image_{str('{:07d}'.format(current_frame))}.jpg", frame)
            current_frame += 1
            if success == False:
                break
        cap.release()
        cv2.destroyAllWindows()
        return fps

    except Exception as ex:
        print(f"{ex}")


def pull_faces_from_images(import_path, extraction_path):
    for (root, dirs, file) in walk(import_path):
        for file_name in file:
            try:
                img = face_recognition.load_image_file(import_path + file_name)
                face_locations = face_recognition.face_locations(img)
                face_count = 0
                skip_frame = False
                for face in face_locations:
                    face_count += 1
                    if face_count > 1:
                        face_count = 0
                        skip_frame = True
                if not skip_frame:
                    top, right, bottom, left = face
                    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(f"{extraction_path + file_name[:-4]}_face.jpg", RGB_img[top:bottom, left:right])
                cv2.destroyAllWindows()
            except Exception as ex:
                print(f"{extraction_path + file_name[:-4]}_face.jpg | {ex}")


# returns a list of coördinats of the faces
def draw_bounding_boxes_on_images(import_path, extraction_path, box_color_list):
    box_color_list_count = 0
    for (root, dirs, file) in walk(import_path):
        for file_name in file:
            try:
                img = face_recognition.load_image_file(import_path + file_name)
                face_locations = face_recognition.face_locations(img)
                face_count = 0
                skip_frame = False
                for face in face_locations:
                    face_count += 1
                    if face_count > 1:
                        face_count = 0
                        skip_frame = True
                if not skip_frame:
                    top, right, bottom, left = face
                    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    color = box_color_list[box_color_list_count]
                    if color == (0, 255, 0):
                        label = 'Real'
                    else:
                        label = 'Fake'
                    cv2.rectangle(RGB_img, (left, top), (right, bottom), color, 2)
                    cv2.putText(RGB_img, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    cv2.imwrite(f"{extraction_path + file_name[:-4]}_boxed_face.jpg", RGB_img)
                    box_color_list_count += 1
                cv2.destroyAllWindows()
            except Exception as ex:
                print(f"{extraction_path + file_name[:-4]}_face.jpg | {ex}")


def compile_bounding_box_video(import_path, extraction_path, fps):
    # get all of the boxed frames
    images = [img for img in listdir(import_path) if img.endswith(".jpg")]
    images.sort()

    if images:  # validation
        # take first frame of the boxed frame dir to get the video resolution
        frame = cv2.imread(path.join(import_path, images[0]))
        height, width, layers = frame.shape

        # Define the video location and dimensions
        # codec = cv2.VideoWriter_fourcc('H', '2', '6', '4')
        codec = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
        video = cv2.VideoWriter(extraction_path + 'prediction.avi', codec, fps, (width, height))
        
        # add the boxed frames to the video
        for img in images:
            video.write(cv2.imread(path.join(import_path, img)))

        cv2.destroyAllWindows()
        video.release()


def delete_files_in_dir(path):
    for (root, dirs, file) in walk(path):
        for file_name in file:
            remove(path + file_name)


def load_image(img_path):
    img = utils.load_img(img_path, target_size=INPUT_SIZE)
    img_tensor = utils.img_to_array(img)  # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor,
                                axis=0)  # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.  # imshow expects values in the range [0, 1]

    return img_tensor


def predict_on_faces(path):
    list_of_predictions = []
    for (root, dirs, file) in walk(path):
        for file_name in file:
            image_face = load_image(path + file_name)
            prediction = model.predict(image_face)
            list_of_predictions.append(prediction)
    return list_of_predictions


def define_box_color(list_of_predictions):
    box_color_list = []
    red = (0, 0, 255)
    green = (0, 255, 0)
    for i in list_of_predictions:
        if i < 0.5:
            box_color_list.append(red)
        else:
            box_color_list.append(green)
    return box_color_list


def get_first_image_from_dir(import_path):
    images = [img for img in listdir(import_path) if img.endswith(".jpg")]
    images.sort()

    return import_path + images[0]
