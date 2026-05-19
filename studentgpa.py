import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
try:
    data = pd.read_csv("ethiopian_student_lifestyle.csv")
except:
    messagebox.showerror("Error", "Dataset not found!")
    exit()

# Drop ID column if exists
if "Student_ID" in data.columns:
    data.drop(columns=["Student_ID"], inplace=True)

# Map categorical variables
stress_map = {"Low": 0, "Moderate": 1, "High": 2}
living_map = {"With Family": 0, "Dormitory": 1}

data["Stress_Level"] = data["Stress_Level"].map(stress_map)
data["Living_Arrangement"] = data["Living_Arrangement"].map(living_map)

# Features affecting GPA (all relevant)
FEATURES = [
    "Study_Hours_Per_Day",
    "Sleep_Hours_Per_Day",
    "Social_Hours_Per_Day",
    "Extracurricular_Hours_Per_Day",
    "Physical_Activity_Hours_Per_Day",
    "Living_Arrangement",
    "Monthly_Budget_ETB",
    "Attendance_Rate_Percent",
    "Stress_Level"
]

X = data[FEATURES]
y = data["GPA"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Random Forest Model (good for GPA prediction)
model = RandomForestRegressor(
    n_estimators=700,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

model.fit(X_train, y_train)


def gpa_class(gpa):
    if gpa < 2.0:
        return "Low"
    elif gpa < 3.0:
        return "Medium"
    else:
        return "High"


def show_confusion_matrix():
    y_pred = model.predict(X_test)

    y_true_class = y_test.apply(gpa_class)
    y_pred_class = [gpa_class(x) for x in y_pred]

    cm = confusion_matrix(
        y_true_class,
        y_pred_class,
        labels=["Low", "Medium", "High"]
    )

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=["Low","Medium","High"],
                yticklabels=["Low","Medium","High"])
    plt.title("Confusion Matrix (GPA Category)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


def show_feature_importance():
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)

    plt.figure(figsize=(8,5))
    plt.barh(np.array(FEATURES)[sorted_idx],
             importance[sorted_idx])
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()


root = tk.Tk()
root.title("GPA Prediction - Random Forest")
root.geometry("600x750")

frame = ttk.Frame(root, padding=20)
frame.pack(fill="both", expand=True)


def create_input(label):
    ttk.Label(frame, text=label).pack(anchor="w")
    entry = ttk.Entry(frame)
    entry.pack(fill="x", pady=5)
    return entry


study_ent = create_input("Study Hours Per Day")
sleep_ent = create_input("Sleep Hours Per Day")
social_ent = create_input("Social Hours Per Day")
phys_ent = create_input("Physical Activity Hours Per Day")
extra_ent = create_input("Extracurricular Hours (Auto)")
budget_ent = create_input("Monthly Budget (ETB)")
att_ent = create_input("Attendance Rate (%)")

ttk.Label(frame, text="Stress Level").pack(anchor="w")
stress_cb = ttk.Combobox(frame, values=["Low","Moderate","High"], state="readonly")
stress_cb.current(1)
stress_cb.pack(fill="x", pady=5)

ttk.Label(frame, text="Living Arrangement").pack(anchor="w")
living_cb = ttk.Combobox(frame, values=["With Family","Dormitory"], state="readonly")
living_cb.current(0)
living_cb.pack(fill="x", pady=5)


def auto_extra(event=None):
    try:
        s1 = float(study_ent.get() or 0)
        s2 = float(sleep_ent.get() or 0)
        s3 = float(social_ent.get() or 0)
        s4 = float(phys_ent.get() or 0)

        total = s1 + s2 + s3 + s4
        extra = 24 - total

        extra_ent.delete(0, tk.END)
        if extra >= 0:
            extra_ent.insert(0, f"{extra:.2f}")
        else:
            extra_ent.insert(0, "0")
    except:
        pass


for e in [study_ent, sleep_ent, social_ent, phys_ent]:
    e.bind("<KeyRelease>", auto_extra)


def validate_attendance(event=None):
    try:
        val = float(att_ent.get())

        # attendance must be between 0 and 100
        if val < 0:
            att_ent.delete(0, tk.END)
            att_ent.insert(0, "0")
        elif val > 100:
            att_ent.delete(0, tk.END)
            att_ent.insert(0, "100")
    except:
        pass


att_ent.bind("<KeyRelease>", validate_attendance)


def predict():
    try:
        s1 = float(study_ent.get())
        s2 = float(sleep_ent.get())
        s3 = float(social_ent.get())
        s4 = float(phys_ent.get())
        extra = float(extra_ent.get())
        budget = float(budget_ent.get())
        att = float(att_ent.get())

        # attendance validation
        if att < 0 or att > 100:
            messagebox.showerror("Error", "Attendance must be between 0 and 100")
            return

        # total hours validation
        if round(s1 + s2 + s3 + s4 + extra, 2) != 24:
            messagebox.showerror("Error", "Total hours must equal 24")
            return

        input_df = pd.DataFrame([{
            "Study_Hours_Per_Day": s1,
            "Sleep_Hours_Per_Day": s2,
            "Social_Hours_Per_Day": s3,
            "Extracurricular_Hours_Per_Day": extra,
            "Physical_Activity_Hours_Per_Day": s4,
            "Living_Arrangement": living_map[living_cb.get()],
            "Monthly_Budget_ETB": budget,
            "Attendance_Rate_Percent": att,
            "Stress_Level": stress_map[stress_cb.get()]
        }])

        input_scaled = scaler.transform(input_df)
        gpa = model.predict(input_scaled)[0]

        # suggestion logic (study & attendance matter)
        if gpa >= 3.6:
            suggestion = "Excellent performance. Keep consistency."
        elif gpa >= 3.1:
            suggestion = "Good performance. Keep studying and attending classes."
        elif gpa >= 2.8:
            suggestion = "Fair. Increase study hours and class attendance."
        else:
            suggestion = "Low GPA predicted. Improve study time and attendance."

        result_lbl.config(text=f"Predicted GPA: {gpa:.2f}")
        suggest_lbl.config(text=suggestion)

    except:
        messagebox.showerror("Error", "Please fill all fields correctly.")


def clear_all():
    for ent in [study_ent, sleep_ent, social_ent, phys_ent,
               extra_ent, budget_ent, att_ent]:
        ent.delete(0, tk.END)

    stress_cb.current(1)
    living_cb.current(0)
    result_lbl.config(text="")
    suggest_lbl.config(text="")


ttk.Button(frame, text="Predict GPA", command=predict).pack(pady=10, fill="x")
ttk.Button(frame, text="View Confusion Matrix", command=show_confusion_matrix).pack(pady=5, fill="x")
ttk.Button(frame, text="View Feature Importance", command=show_feature_importance).pack(pady=5, fill="x")
ttk.Button(frame, text="Clear All", command=clear_all).pack(pady=10, fill="x")

result_lbl = ttk.Label(frame, font=("Arial",16,"bold"))
result_lbl.pack(pady=15)

suggest_lbl = ttk.Label(frame, wraplength=500)
suggest_lbl.pack(pady=5)

root.mainloop()