import smtplib

sm = smtplib.SMTP('smtp.gmail.com', 587) 
sm.starttls()

sm.login("user@gmail.com", "password")

msg = "The best model has been trained successfully."

sm.sendmail("user@gmail.com", "receiver@gmail.com", msg)
sm.quit()
