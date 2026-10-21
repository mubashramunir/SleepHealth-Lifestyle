import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("Sleep Health and Lifestyle Dataset.csv")
data.head()
data.info()

# Convert the cleaned DataFrame into a list of dictionaries
records = data.to_dict(orient="records")

# Merge Sort function (adapted from the task sheet)
def merge_sort(arr, key, reverse=False):
    if len(arr) > 1:
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]

        merge_sort(left_half, key, reverse)
        merge_sort(right_half, key, reverse)

        i = j = k = 0
        while i < len(left_half) and j < len(right_half):
            if reverse:
                condition = left_half[i][key] > right_half[j][key]
            else:
                condition = left_half[i][key] < right_half[j][key]

            if condition:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1

        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1

    return arr

# Sort by Daily Steps, highest to lowest
sorted_records = merge_sort(records, key="Daily Steps", reverse=True)

# Check the top 5 results
for r in sorted_records[:5]:
    print(r["Person ID"], r["Occupation"], r["Daily Steps"])

#MAP    
# Convert Sleep Duration from hours to minutes
data["SleepDurationMinutes"] = list(map(lambda x: x * 60, data["Sleep Duration"]))

# Extract a simplified stress label using map + lambda
data["StressLabel"] = list(map(lambda x: "High" if x >= 7 else "Low", data["Stress Level"]))
print(data[["Sleep Duration", "SleepDurationMinutes", "Stress Level", "StressLabel"]].head())

# FILTER
# Keep only people with high stress (Stress Level > 6)
records = data.to_dict(orient="records")
high_stress = list(filter(lambda x: x["Stress Level"] > 6, records))

print(f"Total records: {len(records)}")
print(f"High stress records: {len(high_stress)}")

#REDUCE
from functools import reduce

# Total Daily Steps per Occupation
steps_by_occupation = [(r["Occupation"], r["Daily Steps"]) for r in records]

total_steps = reduce(
    lambda acc, item: {**acc, item[0]: acc.get(item[0], 0) + item[1]},
    steps_by_occupation,
    {}
)

for occupation, steps in sorted(total_steps.items(), key=lambda x: x[1], reverse=True):
    print(occupation, steps)
    
# Visualization 
# Convert the reduce() dictionary into a DataFrame
df_steps = pd.DataFrame(total_steps.items(), columns=["Occupation", "TotalSteps"])
df_steps = df_steps.sort_values(by="TotalSteps", ascending=False)

plt.figure(figsize=(10, 5))
plt.bar(df_steps["Occupation"], df_steps["TotalSteps"], color="teal")
plt.xticks(rotation=45, ha="right")
plt.xlabel("Occupation")
plt.ylabel("Total Daily Steps")
plt.title("Total Daily Steps by Occupation")
plt.tight_layout()
plt.show()


# Count High vs Low stress using your StressLabel column
stress_counts = data["StressLabel"].value_counts()
plt.figure(figsize=(7, 7))
plt.pie(stress_counts, labels=stress_counts.index, autopct="%1.1f%%", startangle=140, colors=["#B85042", "#69A297"])
plt.title("Proportion of High vs Low Stress")
plt.show()

print(data.isna())
missing_values = data.isnull().sum()
print(missing_values)

print('BMI Category unique values:')
print(data['BMI Category'].value_counts())

print('Blood Pressure sample values:')
print(data['Blood Pressure'].unique()[:10])

print('Gender unique values:')
print(data['Gender'].unique())

print('Sleep Disorder unique values (incl NaN):')
print(data['Sleep Disorder'].value_counts(dropna=False))

print('Occupation unique values:')
print(data['Occupation'].unique())

# Fix inconsistent BMI Category labels
data['BMI Category'] = data['BMI Category'].replace({'Normal Weight': 'Normal'})
print(data['BMI Category'].value_counts())

# Split "126/83" style strings into Systolic and Diastolic
data[['Systolic', 'Diastolic']] = data['Blood Pressure'].str.split('/', expand=True)
data['Systolic'] = data['Systolic'].astype(int)
data['Diastolic'] = data['Diastolic'].astype(int)
data = data.drop(columns=['Blood Pressure'])
print(data[['Systolic', 'Diastolic']].head())

# Missing values here mean "no sleep disorder", not truly unknown
data['Sleep Disorder'] = data['Sleep Disorder'].fillna('None')
print(data['Sleep Disorder'].value_counts())

print(data.isnull().sum())
print(data.dtypes)

numerical_cols = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Stress Level', 'Heart Rate', 'Daily Steps', 'Systolic', 'Diastolic']
print(data[numerical_cols].describe())
# Box Plot for each numerical columns
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
for i, col in enumerate(numerical_cols):
    sns.boxplot(y=data[col], ax=axes[i])
    axes[i].set_title(col)
plt.tight_layout()
plt.show()

#Outlier Handling
Q1 = data['Heart Rate'].quantile(0.25)
Q3 = data['Heart Rate'].quantile(0.75)
IQR = Q3 - Q1
outliers = data[(data['Heart Rate'] < Q1 - 1.5*IQR) | (data['Heart Rate'] > Q3 + 1.5*IQR)]
print(f"Number of Heart Rate outliers: {len(outliers)}")
print(outliers[['Heart Rate', 'Stress Level', 'Occupation']])

# Does stress affect sleep quality?
plt.figure(figsize=(10,6))
sns.scatterplot(data=data, x='Stress Level', y='Quality of Sleep', hue='Sleep Disorder')
plt.title("Stress Level vs Quality of Sleep")
plt.xlabel("Stress Level (1-10)")
plt.ylabel("Quality of Sleep (1-10)")
plt.show()

#How does sleep quality vary by occupation?
plt.figure(figsize=(12,6))
sns.boxplot(data=data, x='Occupation', y='Quality of Sleep')
plt.title("Quality of Sleep by Occupation")
plt.xticks(rotation=45)
plt.show()

# Which BMI categories are linked to sleep disorder?
plt.figure(figsize=(10,6))
sns.countplot(data=data, x='BMI Category', hue='Sleep Disorder')
plt.title("Sleep Disorder Count by BMI Category")
plt.xlabel("BMI Category")
plt.ylabel("Count")
plt.show()

# SLeep Duration By Gender
plt.figure(figsize=(8,6))
sns.boxplot(data=data, x='Gender', y='Sleep Duration', palette=['#69A297', '#B85042'])
plt.title("Sleep Duration by Gender")
plt.ylabel("Sleep Duration (hours)")
plt.show()

#Physical Activity 
plt.figure(figsize=(8,6))
sns.scatterplot(data=data, x='Physical Activity Level', y='Quality of Sleep', hue='Sleep Disorder', alpha=0.7)
plt.title("Physical Activity Level vs Quality of Sleep")
plt.xlabel("Physical Activity Level (minutes/day)")
plt.ylabel("Quality of Sleep (1-10)")
plt.show()

plt.figure(figsize=(8,6))
sns.histplot(data=data, x='Age', bins=15, color='#50808E', kde=True)
plt.title("Age Distribution of Participants")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

data.to_csv("Cleaned Sleep Data.csv", index=False)
