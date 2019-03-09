import serial
#En este caso cuidar cual es el puerto en ejecucion para que funcione desde 
#la terminal, ya que en el SHELL el puerto se llamara diferente
#SHELL =		/dev/tty.wchusbserial1410"
#TERMINAL =		/dev/cu.wchusbserial1410"
arduino = serial.Serial(port="/dev/cu.wchusbserial1410", baudrate=9600)
rawString = arduino.readline()
print((rawString[1:-2]).decode('utf-8'))
#arduino.close()