import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sbn
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from scipy.stats import norm
# dataset import

diabdata = pd.read_csv("K:\\FITA class docs\\AIML\\My_Datasets\\Pima Indians Diabetes Database\\diabetes.csv")

# data inspection and transformation
stat = diabdata.describe()
diamode = diabdata.mode()
diamedian = diabdata.median()
print(diamedian)
print(diamode)
print(stat)
print(diabdata.dtypes)
print(diabdata.isnull().sum())
print(diabdata.duplicated())

# removing skin thickness and insulin columns

columnremove = ['SkinThickness', 'Insulin']
diabdata = diabdata.drop(columns = columnremove)

# column split

refer = diabdata[['Age', 'Pregnancies', 'Glucose', 'BloodPressure', 'BMI', 'DiabetesPedigreeFunction']]
obtained = diabdata['Outcome']

# split train and test data

refer_train, refer_test, obtained_train, obtained_test = train_test_split(refer, obtained, test_size = 0.20, random_state = 42)

# scale standardization

scaler = StandardScaler()
refer_train_scaled = scaler.fit_transform(refer_train)
refer_test_scaled = scaler.transform(refer_test)

# SVC Model Training

model = SVC(kernel = 'poly', random_state = 42)
model.fit(refer_train_scaled, obtained_train)

# Model prediction

obtain_pred = model.predict(refer_test_scaled)

# model metrics evaluation
cm = confusion_matrix(obtained_test, obtain_pred)
print("Confusion Matrix: \n", cm)
print("Accuracy of SVM Classifier model: \n", accuracy_score(obtained_test, obtain_pred))
print("Classification Report: \n", classification_report(obtained_test, obtain_pred))

# Plots

# 1) Count plot

plt.figure(figsize=(12, 10))
ax = sbn.countplot(x='Outcome', data=diabdata, palette='YlOrBr')
ax.set_facecolor('black')
plt.gcf().patch.set_facecolor('black')
ax.bar_label(ax.containers[0], color='white', fontsize=14)
ax.bar_label(ax.containers[1], color='white', fontsize=14)
plt.title("Count plot: Outcome distribution", color='white', fontsize=14)
plt.xlabel("Outcome Class", color='white', fontsize=14)
plt.ylabel("Women Count", color='white', fontsize=14)
ax.tick_params(colors='white', size=20)
legend_elements = [
    Patch(facecolor='none', edgecolor='none', label='0 = Healthy'),
    Patch(facecolor='none', edgecolor='none', label='1 = May have diabetes')
]
ax.legend(handles=legend_elements, loc='upper right', frameon=True, labelcolor='black', fontsize=12)
ax.yaxis.grid(True, color='white', linestyle='--', linewidth=0.6, alpha=0.6)
plt.show()

# 2) Histogram with bell curve
# for age
plt.figure(figsize=(12, 10))
ax = sbn.histplot(diabdata['Age'], bins=15, stat='density', color='#80ed99', edgecolor='black')
ax.set_facecolor('black')
plt.gcf().patch.set_facecolor('black')

# Bell curve
mean_age = diabdata['Age'].mean()
std_age = diabdata['Age'].std()
x = np.linspace(diabdata['Age'].min(), diabdata['Age'].max(), 100)

plt.plot(x, norm.pdf(x, mean_age, std_age), color='white', linewidth=2)
plt.title("Histogram of Age with Bell Curve", color='white', fontsize=16)
plt.xlabel("Age (Years)", color='white', fontsize=14)
plt.ylabel("Density", color='white', fontsize=14)
ax.tick_params(colors='white', labelsize=12)
ax.yaxis.grid(True, color='white', linestyle='--', linewidth=0.6, alpha=0.6)
plt.show()

# for glucose
plt.figure(figsize=(12, 10))
ax = sbn.histplot(diabdata['Glucose'], bins=15, stat='density', color='#ffd166', edgecolor='black')
ax.set_facecolor('black')
plt.gcf().patch.set_facecolor('black')

# Bell curve
mean_age = diabdata['Glucose'].mean()
std_age = diabdata['Glucose'].std()
x = np.linspace(diabdata['Glucose'].min(), diabdata['Glucose'].max(), 100)

plt.plot(x, norm.pdf(x, mean_age, std_age), color='white', linewidth=2)
plt.title("Histogram of Glucose with Bell Curve", color='white', fontsize=16)
plt.xlabel("Glucose mg/dL", color='white', fontsize=14)
plt.ylabel("Density", color='white', fontsize=14)
ax.tick_params(colors='white', labelsize=12)
ax.yaxis.grid(True, color='white', linestyle='--', linewidth=0.6, alpha=0.6)
plt.show()

# confusion matric plot for model performance
plt.figure(figsize=(12, 10))
ax = sbn.heatmap(cm, annot=True, fmt='d', cmap='icefire', cbar=False, annot_kws={'color': 'black', 'fontsize': 20})
ax.set_facecolor('black')
plt.gcf().patch.set_facecolor('black')

ax.set_xticklabels(['Healthy (0)', 'Diabetic (1)'], color='white', fontsize=12)
ax.set_yticklabels(['Healthy (0)', 'Diabetic (1)'], color='white', fontsize=12)
plt.xlabel("Predicted Outcome", color='white', fontsize=14)
plt.ylabel("Actual Outcome", color='white', fontsize=14)
plt.title("Confusion Matrix – SVM Classifier", color='white', fontsize=16)
ax.tick_params(colors='white')
plt.show()

# Fetching User input

print("PIMA INDIAN DIABETES PREDICTION FOR PREGNANT WOMEN")
print("All the data points and predicted outcomes are referred to INDIAN region only")


age = input("Kindly enter maternal women's age: ")
pregnancy = input("Please enter maternity history of the women: ")
glucose = input("Accurately enter the glucose level of the maternal women: ")
bloodpressure = input("Please enter the bottom blood pressure (Diastole) of the maternal women: ")
height_cm = input("\nEnter the Height in centimeters: ")
weight = input("Enter the Weight in Kilograms: ")
#bmi = input("Kindly enter the calculated BMI of the maternal women: ")
dpf = input("Kindly enter the calculated Diabetes pedigree function value of the maternal women: ")

# User input validation

if not age.isdigit() or int(age)<=0:
    print("Kindly enter correct age in whole number")
elif not pregnancy.isdigit() or int(pregnancy)<260:
    print("Kindly enter correct maternal history in whole number")
elif not glucose.replace(","," ",1) or float(glucose)<=0:
    print("Kindly enter accurate glucose level measured")
elif not bloodpressure.isdigit() or int(bloodpressure)<=0:
    print("Kindly enter measured bottom blood pressure value")
elif not height_cm.replace(","," ",1) or float(height_cm)<=0:
    print("Kindly enter accurate height value measured")
elif not weight.replace(","," ",1) or float(weight)<=0:
    print("Kindly enter accurate weight value measured")
elif not dpf.replace(","," ",1) or float(dpf)<=0:
    print("Kindly enter proper diabetes pedigree function value")
else:
    age = int(age)
    pregnancy = int(pregnancy)
    glucose = float(glucose)
    bloodpressure = int(bloodpressure)
    # height conversion for bmi
    height_meter = float(height_cm)/100
    height_m = float(height_meter)
    weight = float(weight)
    # bmi calculation
    bmi = (weight / height_m**2)
    dpf = float(dpf)
    
    # outcome prediction 
    check_diabetic = scaler.transform([[age, pregnancy, glucose, bloodpressure, bmi, dpf]])
    status = model.predict(check_diabetic)
    
    print("Displaying values that are entered")
    print(f"👩 Age is: {age}")
    print(f"🤰 Maternity history of the women: {pregnancy}")
    print(f"🧪 Glucose level: {glucose}")
    print(f"🫀 Bottom bloodpressure: {bloodpressure} mmHg")
    print(f"📏 Height value: {height_m} m")
    print(f"⚖️ Weight value: {weight} Kg")
    print(f"📐 Calculated BMI is: {bmi:.1f} Kg/m^2")
    print(f"📊 Diabetes pedigree function value: {dpf} unitless")

    if status == 1:
        print("⚠️ May result in higher chances for diabetes")

# health related suggestions
        print("🏥 Mandatory care required to improve health during pregnancy: \n")
        print("     📝 Suggested the maternal women to undergo Gestational Diabetes Mellitus screening test")
        print("     🛡 The above test might prevent from further risks for mother and baby")
        print("     💯 Higly Recommended to follow a dietary plan until baby delivery")
        print("     🤏 Limit sugar and refined carbs intake")
        print("     👉 Opt for brown rice, barley, lentils for blood sugar control")
        print("     🍎 Eat fruits in small quantity and limit milk to a small cup amount")
        print("     🚶‍♀️ Practicing Easy and low intensity excercise with normal walking could control blood sugar")
    else:
        print("\n🤰 Congratulations!, you are Healthy and on the right track ✅")
# health related suggestions        
        print("👌 Suggestions to maintain health until delivery")
        print("     💪 Maintain the regular activities")
        print("     💃 Keep your mind active and Take rest when needed")
        print("     🎻 Listen to music for relaxation that helps baby's hearing ability")
        print("     🙅‍♀️ Better to avoid higher food intake amount at a time to maintain the current state")
        print("     🤔 Try to include low intensity exercise and walking with regular activities\n")




'''# References\n

#[1] https://medium.com/data-science/pima-indian-diabetes-prediction-7573698bd5fe\n
#[2] https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database\n
'''

