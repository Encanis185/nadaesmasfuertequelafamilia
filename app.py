import cv2
import os
from flask import Flask, request, render_template, Response
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

####    DEFINICIONES 
classifier_route = 'static/haarcascade_frontalface_default.xml'
codes = {} #Para hacer el diccionario con los codigos
logged_user = ""
####    Denifir la APP de FLASK
app = Flask(__name__)

####    Informacion de la fecha
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

####    Acceso a la webcam // Iniciar video captura
# Seleccionar clasificador
face_detector = cv2.CascadeClassifier(classifier_route)
#Iniciar la camara
try:
    camera = cv2.VideoCapture(0) #Probar primero con camara integrada
except:
    camera = cv2.VideoCapture(1) #Camara "secundaria"

####    Crear directorios    ####
####    ¿Consultar los archivos desde el servidor?    ####
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')


################### FUNCIONES ###########################

"""Esta parte del codigo compara los cuadros con el modelo para determinar si
    la cara presente en los cuadros pertenece a alguna de los presentes en el modelo            
    Si existe coincidencia, dibuja un recuadro alrededor de la cara identificando al usuario 
"""
def identify_person(frame):
    global logged_user
    face_points = extract_faces(frame)
    if np.any(face_points):  # Verificar si hay caras detectadas
        (x, y, w, h) = face_points[0]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
        identified_person = identify_face(face.reshape(1, -1))[0]
        
        code = identified_person.split('_')[1]
        
        cv2.putText(frame, f'{code}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 35), 2, cv2.LINE_AA)

        if code not in codes:
            codes[code] = 1
        else:
            codes[code] += 1
            if codes[code] == 20:
                logged_user = code
                return code

"""
def identify_person(frame):
    global logged_user
    if extract_faces(frame) != ():
        

        (x, y, w, h) = extract_faces(frame)[0]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
        identified_person = identify_face(face.reshape(1, -1))[0]
        
        code = identified_person.split('_')[1]
        
        cv2.putText(frame, f'{code}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 35), 2, cv2.LINE_AA)

        if code not in codes:
            codes[code] = 1
        else:
            codes[code] += 1
            if codes[code] == 20:
                logged_user = code
                return code
"""


def get_logged_id(code):
    return code

    
#Genera el cuadro de video en el navegador :)
def gen_frames():
    global codes
    codes = {}
    while True:
        success, frame = camera.read()  # Lee la informacion de la camara
        if not success:
            break
        else:
            # Devolver/Obtener el codigo de estudiante
            ############
            # Validando que sea el usuario seleccionado
            print(identify_person(frame))
            print(logged_user)
            
            #return logged_user 
            
            #########
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 

#### Obtener total de usuarios registrados, basado en la cantidad de carpetas
def totalreg():
    return len(os.listdir('static/faces'))

#### Extrae las caras de las imagenes
def extract_faces(img):
    if img.size != 0:  # Verificar si la matriz no está vacía
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.3, 5)
        return face_points
    else:
        return []

#### Identifica las caras utiliando un modelo de ML
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


#### Entrena el modelo de KNN con las caras  y nombres presentes en los folders
def train():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')


#### Obtener informacion de la asistecia del archivo csv

def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l


#### Agregar asistencia a un usuario

def add_attendance(name):
    #username = name.split('_')[0]
    #user_code = name.split('_')[1] #Codigo de estudiante
    #current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')


################## ROUTING FUNCTIONS #########################

#### Pagina principal
@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


#### Llamado al hacer click en tomar asistencia
""" Ignora de momento esto"""
@app.route('/start', methods=['GET'])
def start():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', totalreg=totalreg(), datetoday2=datetoday2,
                               mess='There is no trained model in the static folder. Please add a new face to continue.')
    ret = True

    while ret:
        ret, frame = camera.read()
        identify_person(frame)

        if extract_faces(frame) != ():
            (x, y, w, h) = extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 00), 2)
            face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person)
            cv2.putText(frame, f'{identified_person}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 35), 2, cv2.LINE_AA)

        cv2.imshow('Attendance', frame)
        
        if cv2.waitKey(1) == 27:
            break
    camera.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2)


#### Agregar un nuevo alumno
@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/' + newusername + '_' + str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    
    i, j = 0, 0
    while 1:
        _, frame = camera.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/10', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2,
                        cv2.LINE_AA)
            if j % 10 == 0:
                name = newusername + '_' + str(i) + '.jpg'
                cv2.imwrite(userimagefolder + '/' + name, frame[y:y + h, x:x + w])
                i += 1
            j += 1
        if j == 100:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    camera.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

#### :)
if __name__ == '__main__':
    
    app.run(debug=True)

