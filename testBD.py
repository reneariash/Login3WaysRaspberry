import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="1234567890",
  database="bd_3_factores"
)

mycursor = mydb.cursor()

mycursor.execute("SELECT * FROM usuarios")

myresult = mycursor.fetchall()

for x in myresult:
  print(x)