import glob
from random import random

from flask import Flask, request, render_template, redirect,jsonify
import os
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import cv2
import base64
import io
import pandas as pd


model = load_model('fracture_model.h5')

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('Index.html')


app.config['IMAGE_UPLOADS'] = "static\\pics\\"


@app.route("/uploadimage", methods=['POST', 'GET'])
def image():
    return render_template('uploadimage.html')






# test_image = image.load_img(test_img, target_size=(64, 64))
    # test_image = image.img_to_array(test_image)
    # test_image = np.expand_dims(test_image, axis=0)
    # result = model.predict(test_image)
    # if result[0][0] == 0:
    #     return render_template('report.html',prediction='FRACTURE IS DETECTED!!')
    #
    # else:
    #     return render_template('report.html',prediction='NO FRACTURE IS DETECTED!!'



@app.route('/loggedin', methods=['POST', 'GET'])
def save_credentials():
    save_credentials.username = request.form['username']
    save_credentials.password = request.form['password']
    credentials = pd.read_csv('pasword detail.csv')
    passwords = list(credentials.Password)
    usernames = list(credentials.User_Name)
    if save_credentials.username in usernames and save_credentials.password in passwords:
        return render_template('PatientDetails.html')
    else:
        return render_template('Login.html', message='Either password or username not correct! or try signing up')


@app.route('/signup', methods=['POST', 'GET'])
def upload_credentials():
    upload_credentials.username = request.form['uname']
    upload_credentials.password = request.form['pword']
    new_details = pd.DataFrame({"User_Name": [upload_credentials.username],
                                 "Password": [upload_credentials.password]})
    credentials = pd.read_csv('pasword detail.csv')
    credentials = credentials.append(new_details, ignore_index=True)
    credentials.to_csv("pasword detail.csv", index=False)
    return render_template('Login.html')



@app.route('/cameraacess')
def camera():
    return render_template("camera.html")



@app.route('/submit',methods=['POST'])
def submit():

    image = request.form['image']
    print(image)
    with open(os.path.join(app.config['IMAGE_UPLOADS'], "see.jpg"), "wb") as fh:
        fh.write(base64.b64decode(str(image.split(',')[1])))

    return ""


@app.route("/PatientDetails", methods=['POST', 'GET'])
def details():
    return render_template('PatientDetails.html')


@app.route("/Logins", methods=['POST', 'GET'])
def logins():
    return render_template('Login.html')


@app.route("/signups", methods=['POST', 'GET'])
def signups():
    return render_template('signup.html')


name = ''
age = ''
gender = 0

patient_details = []
@app.route("/submitdetails", methods=['POST', 'GET'])
def getdetails():
    getdetails.name = request.form['name']
    getdetails.age = request.form['age']
    getdetails.gender = request.form['gender']

    patient_details.append(getdetails.name)
    patient_details.append(getdetails.age)
    patient_details.append(getdetails.gender)


    return render_template("PatientDetails.html")


@app.route("/makereport", methods=['POST','GET'])
def createreport():
    if request.method == "POST":

        if request.files:

            test_img = request.files["image"]

            test_img.save(os.path.join(app.config['IMAGE_UPLOADS'], "see.jpg"))

            return redirect(request.url)

    # Load Yolo
    net = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_testing.cfg")

    # Name custom object
    classes = [""]

    # Images path
    images_path = glob.glob(r"static\pics\see.jpg")

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Insert here the path of your images

    # loop through all the images
    for img_path in images_path:
        # Loading image
        img = cv2.imread(img_path)
        img = cv2.resize(img, None, fx=0.4, fy=0.4)
        height, width, channels = img.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:
                    # Object detected
                    print(class_id)
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        print(indexes)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, 3, color, 2)

        cv2.imwrite(r"static\pics\result.jpg", img)

        if len(boxes)==0:
            createreport.found = "NO FRACTURE DETECTED"
        else:
            createreport.found = "FRACTURE DETECTED"

    return render_template('report.html', name1=getdetails.name, age1=getdetails.age, sex1=getdetails.gender, found1=createreport.found)


@app.route("/update", methods=['POST','GET'])
def update ():
    patient_details_ = pd.read_csv('saved Details.csv')
    new_patient = pd.DataFrame({"Patient name": [getdetails.name],
                                "Patient age": [getdetails.age],
                                "Patient Gender": [getdetails.gender],
                                "Result": [createreport.found]})
    patient_details_ = patient_details_.append(new_patient, ignore_index=True)
    patient_details_.to_csv("Saved details.csv", index=False)
    return render_template('final submission.html')


if __name__ == '__main__':
    app.run()
