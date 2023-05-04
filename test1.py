from selenium import webdriver
import time

driver = webdriver.Chrome("C:\\Users\\himaj\\Downloads\\chromedriver_win32\\chromedriver")
driver.get("http://127.0.0.1:5000/")
driver.execute_script("window.scrollTo(0, 200)") 
gender=driver.find_element("id","gender")
gender.send_keys("1")
age=driver.find_element("id","age")
age.send_keys("67")
hypertension=driver.find_element("id","hypertension")
hypertension.send_keys("0")
heart_disease=driver.find_element("id","heart_disease")
heart_disease.send_keys("1")
ever_married=driver.find_element("id","ever_married")
ever_married.send_keys("1")
work_type=driver.find_element("id","work_type")
work_type.send_keys("2")
Residence_type=driver.find_element("id","Residence_type")
Residence_type.send_keys("1")
avg_glucose_level=driver.find_element("id","avg_glucose_level")
avg_glucose_level.send_keys("228.69")
bmi=driver.find_element("id","bmi")
bmi.send_keys("36.6")
smoking_status=driver.find_element("id","smoking_status")
smoking_status.send_keys("1")
time.sleep(5)
driver.find_element("id","submit").click()
time.sleep(5)
driver.quit()
