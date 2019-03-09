from tkinter import *
#import Forecast
import threading
import tkinter as tk
import tkinter.messagebox as tm
import mysql.connector
import serial
import time
import cv2, os
import numpy as np
from PIL import Image


class Frames(object):

    def __init__(self):
        pass

    def main_frame(self, root):
        root.title('WeatherMe')
        self.label_username = tk.Label(root, text="Username")
        self.label_password = tk.Label(root, text="Password")
        self.entry_username = tk.Entry(root)
        self.entry_password = tk.Entry(root, show="*")
        self.label_username.grid(row=0)
        self.label_password.grid(row=1)
        self.entry_username.grid(row=0, column=1)
        self.entry_password.grid(row=1, column=1)
        self.logbtn = tk.Button(root, text="Iniciar Sesion", command=self._login_btn_clicked)
        self.logbtn.grid(columnspan=2)
        self.singbtn = tk.Button(root, text="Registrarse", command=self.singup)
        self.singbtn.grid(columnspan=2)

    def _login_btn_clicked(self):
        print(self.entry_username.get())
        username = self.entry_username.get()
        password = self.entry_password.get()
        mydb = mysql.connector.connect(
          host="localhost",
          user="raspbd",
          passwd="1234567890",
          database="bd_3_factores"
        )
        mycursor = mydb.cursor()
        sql = "SELECT * FROM usuarios WHERE usuario ='"+username+"'"
        print("username: " + username)
        mycursor.execute(sql)
        self.myresult1 = mycursor.fetchall()
        if not self.myresult1:
            print("Usuario no encontrado")
            tm.showerror("Inicio de sesión","Usuario y/o contraseña no validos")
        else:# si el usuario existe compara contraseñas
            if password == self.myresult1[0][4]:
                # Create Local Binary Patterns Histograms for face recognization
                recognizer = cv2.face.LBPHFaceRecognizer_create()#LBPHFaceRecognizer_create
                # Load the trained mode
                recognizer.read('trainer/trainer.yml')
                # Load prebuilt model for Frontal Face
                cascadePath = "haarcascade_frontalface_default.xml"
                # Create classifier from prebuilt model
                faceCascade = cv2.CascadeClassifier(cascadePath);
                # Set the font style
                font = cv2.FONT_HERSHEY_SIMPLEX
                # Initialize and start the video frame capture
                cam = cv2.VideoCapture(0)
                # Loop
                verFacePos = 0
                verFaceNeg = 0
                while True:
                    # Read the video frame
                    ret, im =cam.read()
                    # Convert the captured frame into grayscale
                    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
                    # Get all face from the video frame
                    faces = faceCascade.detectMultiScale(gray, 1.2,5)
                    # For each face in faces
                    for(x,y,w,h) in faces:
                        # Create rectangle around the face
                        cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)
                        # Recognize the face belongs to which ID
                        Id,conf = recognizer.predict(gray[y:y+h,x:x+w])
                        #print(Id)
                        print(conf)
                        # Check the ID if exist 
                        if(Id == self.myresult1[0][0] and conf <= 63):#myresult[0][0]--> id
                            Id = self.myresult1[0][1]#myresult[0][1] --> nombre
                            verFacePos += 1
                        #If not exist, then it is Unknown
                        else:
                            print(Id)
                            Id = "Desconocido"
                            verFaceNeg += 1
                            print("negsss: ", verFaceNeg)
                        #Put text describe who is in the picture
                        cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
                        cv2.putText(im, str(Id), (x,y-40), font, 2, (255,255,255), 3)
                    # Display the video frame with the bounded rectangle
                    cv2.imshow('im',im) 
                    print("pos: ", verFacePos)
                    print("neg: ", verFaceNeg)
                    # If 'q' is pressed, close program
                    if ((cv2.waitKey(10) & 0xFF == ord('q')) or verFacePos >= 10 or verFaceNeg >= 40):
                        break
                # Stop the camera
                cam.release()
                # Close all windows
                cv2.destroyAllWindows()
                if verFacePos >= 10:
                    self.tareas1()
                else:
                    print("Reconocimiento facial fallado...")
                    tm.showerror("Error", "Reconocimiento facial no reconocido")
            else:
                tm.showerror("Login error", "Incorrect username")

    def tareas1(self):
        hilo1 = threading.Thread(target=self.aviso_arduino1)
        hilo2 = threading.Thread(target=self.arduino_cap1)
        hilo1.start()
        hilo2.start()

    def aviso_arduino1(self):
        self.cap_win = Toplevel()
        self.cap_win.title('terecera')
        self.cap_win.geometry('600x150')
        self.label_username1 = tk.Label(self.cap_win, text="Aproxime una tarjeta RFID...").pack()
        print("antes de capturar a punto...")

    def arduino_cap1(self):
        arduino = serial.Serial(port="/dev/ttyUSB0", baudrate=9600)
        rawString = arduino.readline()
        codigo = (rawString[1:-2]).decode('utf-8')
        print("Tarjeta capturada: ", codigo)
        arduino.close()
        if (codigo == self.myresult1[0][3]): #myresult[0][3] --> rfid
            print("RFID Exitoso!!!")
            tm.showinfo("Inicio de sesión","Inicio de sesión exitoso!!!")
        else:
            print("RFID falso")
            tm.showerror("Inicio de sesión","Tarjeta RFID incorrecta")
        self.cap_win.destroy()


    def singup(self):
        self.result = Toplevel()
        self.result.title('segunda')
        self.query = StringVar()

        def verificar_username1(event):
            if self._username1.get() == '':
                print("ingrese")
                self.nota_username1['text']="Ingrese un nombre de usuario"
                self._username1.focus()
            else:
                mydb = mysql.connector.connect(
                  host="localhost",
                  user="raspbd",
                  passwd="1234567890",
                  database="bd_3_factores"
                )
                mycursor = mydb.cursor()
                sql = "SELECT usuario FROM usuarios WHERE usuario ='"+self._username1.get()+"'"
                mycursor.execute(sql)
                myresult = mycursor.fetchall()
                if not myresult:
                    print("Usuario disponible")
                    self.nota_username1['text'] = 'Correcto!!!'
                else:# si el usuario existe compara contraseñas      
                    print("El usuario ya existe")    
                    self.nota_username1['text'] = 'Usuario no disponible!!!'
                    self._username1.focus()  

            
        self.label_username1 = tk.Label(self.result, text="Username: ")
        self.label_nombre = tk.Label(self.result, text="Nombre: ")
        self.label_password = tk.Label(self.result, text="Password: ")
        self.label_password2 = tk.Label(self.result, text="Repetir password: ")
        self.label_tarjeta= tk.Label(self.result, text="Ingreso de tarjeta: ")
        self.label_facial = tk.Label(self.result, text="Reconocimiento facial: ")        
        self._username1 = tk.Entry(self.result)
        self._nombre = tk.Entry(self.result)
        self._password = tk.Entry(self.result, show="*")
        self._password2 = tk.Entry(self.result, show="*")        
        self.btn_tarjeta = tk.Button(self.result, text="Iniciar captura de tarjeta", command=self.tareas)        
        self.btn_facial = tk.Button(self.result, text="Iniciar reconocimiento facial", command=self.reconocimiento_facial_reg)
        self.nota_username1 = tk.Label(self.result, text="...")
        self.nota_nombre = tk.Label(self.result, text="...")
        self.nota_password = tk.Label(self.result, text="...")
        self.nota_password2 = tk.Label(self.result, text="...")
        self.nota_tarjeta = tk.Label(self.result, textvariable=self.query, text="...")
        self.nota_facial = tk.Label(self.result, text="...")
        self.btn_reg_usuario = tk.Button(self.result, text="Registrar nuevo usuario", command=self.registrar_usuario)
        self.label_username1.grid(row=0, column=0, sticky=tk.E)
        self.label_nombre.grid(row=1, column=0, sticky=tk.E)
        self.label_password.grid(row=2, column=0, sticky=tk.E)
        self.label_password2.grid(row=3, column=0, sticky=tk.E)
        self.label_tarjeta.grid(row=4, column=0, sticky=tk.E)
        self.label_facial.grid(row=5, column=0, sticky=tk.E)
        self._username1.grid(row=0, column=1)
        self._nombre.grid(row=1, column=1)
        self._password.grid(row=2, column=1)
        self._password2.grid(row=3, column=1)
        self.btn_tarjeta.grid(row=4, column=1, sticky=tk.E)
        self.btn_facial.grid(row=5, column=1, sticky=tk.E)       
        self.nota_username1.grid(row=0, column=2, sticky=tk.E)
        self.nota_nombre.grid(row=1, column=2, sticky=tk.E)
        self.nota_password.grid(row=2, column=2, sticky=tk.E)
        self.nota_password2.grid(row=3, column=2, sticky=tk.E)
        self.nota_tarjeta.grid(row=4, column=2, sticky=tk.E)
        self.nota_facial.grid(row=5, column=2, sticky=tk.E)
        self.btn_reg_usuario.grid(row=7, column=2, sticky=tk.E)
        self._username1.focus()
        self._username1.bind('<FocusOut>', verificar_username1)
     
    def registrar_usuario(self):
        if self._nombre.get() == '':
            self.nota_nombre['text'] = 'Ingrese un nombre'
        else:
            self.nota_nombre['text'] = 'Correcto!!!'

        if self._password.get() != '' and self._password2.get() != '' and self._password.get() == self._password2.get():
            self.nota_password['text'] = 'Correcto!!!'
            self.nota_password2['text'] = 'Correcto!!!'
        else:
            self.nota_password['text'] = 'Error en contraseña!!!'

        if self.nota_nombre['text'] == 'Correcto!!!':
            if self.nota_username1['text'] == 'Correcto!!!':
                if self.nota_password['text'] == 'Correcto!!!':
                    if self.nota_password2['text'] == 'Correcto!!!':
                        if self.nota_facial['text'] == 'Correcto!!!':
                            if self.nota_tarjeta['text'] != '':
                                if self.nota_username1['text'] == 'Correcto!!!':
                                    mydb = mysql.connector.connect(
                                              host="localhost",
                                              user="raspbd",
                                              passwd="1234567890",
                                              database="bd_3_factores"
                                            )
                                    mycursor = mydb.cursor()
                                    sql = "INSERT INTO usuarios (nombre, usuario, rfid, contrasena) VALUES ('"+self._nombre.get()+"', '"+self._username1.get()+"', '"+self.nota_tarjeta['text']+"', '"+self._password.get()+"')"
                                    mycursor.execute(sql)
                                    mydb.commit()
                                    print("Usuario registrado correctamente")
                                    tm.showinfo("Registro","Registro exitoso!!!")
                                    self.result.destroy()
        else:
            print('No se han completado los datos')
        print('registrando usurio')

    def reconocimiento_facial_reg(self):
        mydb = mysql.connector.connect(
                  host="localhost",
                  user="raspbd",
                  passwd="1234567890",
                  database="bd_3_factores"
                )
        mycursor = mydb.cursor()
        sql = "SELECT id FROM usuarios ORDER BY id DESC LIMIT 1;"
        mycursor.execute(sql)
        myresult = mycursor.fetchall()
        num_users = 0
        if not myresult:
            num_users = 0
        else:# si el usuario existe compara contraseñas      
            num_users = myresult[0][0]

        print("111")
        if(self.btn_facial["text"]=="Iniciar reconocimiento facial"):
            print("222")
            self.btn_facial.configure(text="Parar reconocimiento facial")
            # Start capturing video 
            vid_cam = cv2.VideoCapture(0)
            # Detect object in video stream using Haarcascade Frontal Face
            face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            # For each person, one face id
            face_id = num_users + 1
            # Initialize sample face image
            count = 0
            # Start looping
            while(True):
                # Capture video frame
                _, image_frame = vid_cam.read()
                # Convert frame to grayscale
                gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
                # Detect frames of different sizes, list of faces rectangles
                faces = face_detector.detectMultiScale(gray, 1.3, 5)
                # Loops for each faces
                for (x,y,w,h) in faces:
                    # Crop the image frame into rectangle
                    cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)
                    # Increment sample face image
                    count += 1
                    print('Foto #: ',count)
                    self.nota_facial['text'] = 'Foto #: ',count
                    # Save the captured image into the datasets folder
                    cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
                    # Display the video frame, with bounded rectangle on the person's face
                    cv2.imshow('frame', image_frame)
                # To stop taking video, press 'q' for at least 100ms
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
                # If image taken reach 100, stop taking video
                elif count>100:
                    print('Termino el reconocimiento facial')
                    print('Iniciando red neuronal')
                    break
            # Stop video
            vid_cam.release()
            # Close all started windows
            cv2.destroyAllWindows()
            self.nota_facial['text'] = 'Generando red neuronal...'
            if count>100:          
                #mandar a relalizar red neuronal
                path = 'dataset'

                # Create Local Binary Patterns Histograms for face recognization
                recognizer = cv2.face.LBPHFaceRecognizer_create()

                # Using prebuilt frontal face training model, for face detection
                detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

                # Create method to get the images and label data
                def getImagesAndLabels(path):

                    # Get all file path
                    imagePaths = [os.path.join(path,f) for f in os.listdir(path)] 
                    
                    # Initialize empty face sample
                    faceSamples=[]
                    
                    # Initialize empty id
                    ids = []

                    # Loop all the file path
                    for imagePath in imagePaths:

                        # Get the image and convert it to grayscale
                        PIL_img = Image.open(imagePath).convert('L')

                        # PIL image to numpy array
                        img_numpy = np.array(PIL_img,'uint8')

                        # Get the image id
                        id = int(os.path.split(imagePath)[-1].split(".")[1])
                        #print(id)

                        # Get the face from the training images
                        faces = detector.detectMultiScale(img_numpy)

                        # Loop for each face, append to their respective ID
                        for (x,y,w,h) in faces:

                            # Add the image to face samples
                            faceSamples.append(img_numpy[y:y+h,x:x+w])

                            # Add the ID to IDs
                            ids.append(id)

                    # Pass the face array and IDs array
                    return faceSamples,ids


                # Get the faces and IDs
                faces,ids = getImagesAndLabels('dataset')

                # Train the model using the faces and IDs
                recognizer.train(faces, np.array(ids))

                # Save the model into trainer.yml
                recognizer.write('trainer/trainer.yml')
                print('Se genero red neuronal')
                self.nota_facial['text'] = 'Correcto!!!'
            else:
                self.nota_facial['text'] = 'Falla'
        else:
            self.btn_facial.configure(text="Iniciar reconocimiento facial")   

    def close_windows(self):
        #print(self.lbl['text'])
        self.master.destroy()

    def tareas(self):
        hilo1 = threading.Thread(target=self.aviso_arduino)
        hilo2 = threading.Thread(target=self.arduino_cap)
        hilo1.start()
        hilo2.start()

    def parar_arduino(self):
        self.arduino.close()

    
    def aviso_arduino(self):
        self.cap_win = Toplevel()
        self.cap_win.title('terecera')
        self.cap_win.geometry('600x150')
        self.label_username1 = tk.Label(self.cap_win, text="Aproxime una tarjeta RFID...").pack()
        print("antes de capturar a punto...")

        def cambiar_nota():
            self.label_username1['text'] = 'Tarjerta capurada exitosamente'

    def arduino_cap(self):
        print('otra')
        self.arduino = serial.Serial(port="/dev/ttyUSB0", baudrate=9600)
        print("antes de capturar a punto...")
        rawString = self.arduino.readline()
        codigo = (rawString[1:-2]).decode('utf-8')
        print("Tarjeta capturada: ", codigo)
        self.arduino.close()
        #self.label_username1['text'] = "Tarjeta capturada exitosamente!!!"
        self.cambia_tarjeta(codigo)

        self.cap_win.destroy()

    def cambia_tarjeta(self, txt):
        self.query.set(txt)

root = Tk()
app = Frames()
app.main_frame(root)
root.mainloop()
