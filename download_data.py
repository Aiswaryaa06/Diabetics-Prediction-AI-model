import urllib.request
import csv

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
output_file = "diabetes.csv"

# Download the file
urllib.request.urlretrieve(url, "temp_data.csv")

# Add headers
headers = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]

with open("temp_data.csv", "r") as infile, open(output_file, "w", newline="") as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    writer.writerow(headers)
    for row in reader:
        writer.writerow(row)

import os
os.remove("temp_data.csv")
print(f"Data successfully saved to {output_file}")
