import requests

all_records = []
page = 1

while len(all_records) < 12000:
    url = (
        f"https://world.openfoodfacts.org/api/v2/search?"
        f"categories=chocolates&fields=code,product_name,brands,nutriments&page_size=100&page={page}"
    )
    response = requests.get(url)
    data = response.json()
    products = data.get("products", [])

    if not products:
        print(f"Page {page} returned no products. Stopping early.")
        break

    all_records.extend(products)
    print(f"Page {page}: {len(products)} products collected. Total: {len(all_records)}")
    page += 1

print(f"\nFinished collecting {len(all_records)} chocolate product records.")


import pandas as pd

# Convert the list of dictionaries into a DataFrame
df = pd.DataFrame(all_records)

df


import pandas as pd

# Main DataFrame with general product info
main_df = pd.DataFrame(all_records)

# Nutriments DataFrame extracted from the nested 'nutriments' field
nutriments_df = pd.json_normalize(all_records, record_path=None, sep='_', meta=['code'])

# Keep only the 'nutriments' part
nutriments_df = pd.DataFrame(main_df['nutriments'].tolist())
nutriments_df['code'] = main_df['code']  # Add product code for linking

nutriments_df


import pandas as pd

# Assuming you already have all_records from your API collection
products_df = pd.DataFrame(all_records)

# Expand the 'nutriments' dictionary into its own DataFrame
nutriments_df = pd.DataFrame(products_df['nutriments'].tolist())

# Add product code to link back
nutriments_df['code'] = products_df['code']

# Select only the fields you care about
selected_fields = [
    'energy-kcal_value',
    'energy-kj_value',
    'carbohydrates_value',
    'sugars_value',
    'fat_value',
    'saturated-fat_value',
    'proteins_value',
    'fiber_value',
    'salt_value',
    'sodium_value',
    'nova-group',
    'nutrition-score-fr',
    'fruits-vegetables-nuts-estimate-from-ingredients_100g',
    'code'
]

nutriments_selected = nutriments_df[selected_fields]

# Merge on 'code' to combine product info with nutrient values
merged_df = products_df.merge(nutriments_selected, on='code', how='left')

merged_df

merged_df['sugar_to_carb_ratio'] = merged_df['sugars_value'] / merged_df['carbohydrates_value']

def classify_calories(kcal):
    if pd.isna(kcal):
        return 'Unknown'
    elif kcal < 100:
        return 'Low'
    elif kcal < 300:
        return 'Moderate'
    else:
        return 'High'

merged_df['calorie_category'] = merged_df['energy-kcal_value'].apply(classify_calories)

merged_df['is_ultra_processed'] = merged_df['nova-group'].apply(lambda x: 'Yes' if x == 4 else 'No')

merged_df

# Count missing values per column
missing_counts = merged_df.isnull().sum()

# Display only columns with missing values
missing_summary = missing_counts[missing_counts > 0].sort_values(ascending=False)

missing_summary


# Count missing values per column
missing_counts = merged_df.isnull().sum()

# Show only columns with missing values
missing_summary = missing_counts[missing_counts > 0].sort_values(ascending=False)
print("Columns with missing values:\n", missing_summary)

for col in merged_df.select_dtypes(include='number').columns:
    if merged_df[col].isnull().sum() > 0:
        merged_df[col].fillna(merged_df[col].median(), inplace=True)
        
for col in merged_df.select_dtypes(include='object').columns:
    if merged_df[col].isnull().sum() > 0:
        merged_df[col].fillna("Unknown", inplace=True)
        
print("Remaining missing values:", merged_df.isnull().sum().sum())

merged_df

print(merged_df.shape)        # Rows and columns
print(merged_df.info())       # Data types and non-null counts
print(merged_df.describe())   # Summary stats for numeric columns


import matplotlib.pyplot as plt
import seaborn as sns
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.histplot(merged_df['energy-kcal_value'], bins=30, ax=axes[0], color='orange')
axes[0].set_title("Energy (kcal) Distribution")

sns.histplot(merged_df['sugars_value'], bins=30, ax=axes[1], color='red')
axes[1].set_title("Sugar Distribution")

sns.histplot(merged_df['sugar_to_carb_ratio'], bins=30, ax=axes[2], color='purple')
axes[2].set_title("Sugar-to-Carb Ratio Distribution")

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.countplot(x='calorie_category', data=merged_df, ax=axes[0], palette='Blues')
axes[0].set_title("Calorie Category Distribution")

sns.countplot(x='sugar_category', data=merged_df, ax=axes[1], palette='Reds')
axes[1].set_title("Sugar Category Distribution")

sns.countplot(x='nova-group', data=merged_df, ax=axes[2], palette='Greens')
axes[2].set_title("NOVA Group Distribution")

plt.tight_layout()
plt.show()


sns.countplot(x='is_ultra_processed', data=merged_df, palette='Set2')
plt.title("Ultra-Processed vs Minimally Processed Products")
plt.show()

sns.scatterplot(x='energy-kcal_value', y='sugars_value', data=merged_df, hue='calorie_category')
plt.title("Calories vs Sugar")
plt.show()

sns.scatterplot(x='energy-kcal_value', y='nova-group', data=merged_df)
plt.title("Energy vs NOVA Group")
plt.show()

import matplotlib.pyplot as plt

# Count the number of products in each calorie category
calorie_counts = merged_df['calorie_category'].value_counts()

# Plot the pie chart
plt.figure(figsize=(8, 8))
plt.pie(
    calorie_counts,
    labels=calorie_counts.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=['#A2D2FF', '#FFAFCC', '#BDB2FF', '#FFC6FF']
)
plt.title("Distribution of Chocolate Products by Calorie Category")
plt.axis('equal')  # Ensures the pie is a circle
plt.show()

numeric_cols = merged_df.select_dtypes(include='number')
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Between Nutritional Variables")
plt.show()


top_brands = (
    merged_df.groupby('brands')[['sugars_value', 'energy-kcal_value']]
    .mean()
    .sort_values(by='sugars_value', ascending=False)
    .head(10)
)

top_brands.plot(kind='bar', figsize=(12, 6), title="Top 10 Brands by Avg Sugar & Calories")
plt.ylabel("Average per 100g")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


import mysql.connector

# Replace with your TiDB connection details
connection = mysql.connector.connect(
        host = "gateway01.ap-southeast-1.prod.aws.tidbcloud.com",
        port = 4000,
        user = "4JX5zZKcrMbbGKn.root",
        password = "ali9zRfvEMFC0mIS",
        database = "Chococrunch_analysis",
)
mycursor = connection.cursor(buffered=True)


mycursor.execute("CREATE DATABASE Chococrunch_analysis;")

mycursor.execute("""
    CREATE TABLE chococrunch_analysis.product_info (
        code VARCHAR(50) PRIMARY KEY,
        product_name TEXT,
        brands TEXT
    );
""")

mycursor.execute("""
    CREATE TABLE chococrunch_analysis.nutrient_info (
        code VARCHAR(50),
        energy_kcal REAL,
        energy_kj REAL,
        carbohydrates REAL,
        sugars REAL,
        fat REAL,
        saturated_fat REAL,
        proteins REAL,
        fiber REAL,
        salt REAL,
        sodium REAL,
        nova_group INTEGER,
        nutrition_score_fr INTEGER,
        fruits_vegetables_nuts_estimate REAL,
        FOREIGN KEY (code) REFERENCES chococrunch_analysis.product_info(code)
    );
""")


mycursor.execute("""
    CREATE TABLE chococrunch_analysis.derived_metrics (
        code VARCHAR(50),
        sugar_to_carb_ratio REAL,
        calorie_category VARCHAR(20),
        sugar_category VARCHAR(20),
        is_ultra_processed VARCHAR(5),
        FOREIGN KEY (code) REFERENCES chococrunch_analysis.product_info(code)
    );
""")


merged_df.to_csv('Chococrunch_analysis,py', index=False)
merged_df



df_product = merged_df[['code', 'product_name', 'brands']].dropna(subset=['code']).drop_duplicates()

df_nutrient = merged_df[[
    'code', 'energy-kcal_value', 'energy-kj_value', 'carbohydrates_value',
    'sugars_value', 'fat_value', 'saturated-fat_value', 'proteins_value',
    'fiber_value', 'salt_value', 'sodium_value', 'nova-group',
    'nutrition-score-fr', 'fruits-vegetables-nuts-estimate-from-ingredients_100g'
]].dropna(subset=['code']).drop_duplicates()

df_nutrient.columns = [
    'code', 'energy_kcal', 'energy_kj', 'carbohydrates', 'sugars', 'fat',
    'saturated_fat', 'proteins', 'fiber', 'salt', 'sodium', 'nova_group',
    'nutrition_score_fr', 'fruits_vegetables_nuts_estimate'
]

df_metrics = merged_df[[
    'code', 'sugar_to_carb_ratio', 'calorie_category',
    'sugar_category', 'is_ultra_processed'
]].dropna(subset=['code']).drop_duplicates()


data = df_product[['code', 'product_name', 'brands']].dropna().values.tolist()

mycursor.executemany("""
    INSERT IGNORE INTO chococrunch_analysis.product_info (code, product_name, brands)
    VALUES (%s, %s, %s)
""", data)

connection.commit()
print(f"Bulk inserted {len(data)} rows into product_info")


data_nutrient = df_nutrient[[
    'code', 'energy_kcal', 'energy_kj', 'carbohydrates', 'sugars', 'fat',
    'saturated_fat', 'proteins', 'fiber', 'salt', 'sodium',
    'nova_group', 'nutrition_score_fr', 'fruits_vegetables_nuts_estimate'
]].dropna(subset=['code']).values.tolist()

mycursor.executemany("""
    INSERT IGNORE INTO chococrunch_analysis.nutrient_info (
        code, energy_kcal, energy_kj, carbohydrates, sugars, fat,
        saturated_fat, proteins, fiber, salt, sodium,
        nova_group, nutrition_score_fr, fruits_vegetables_nuts_estimate
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
""", data_nutrient)

connection.commit()
print(f" Bulk inserted {len(data_nutrient)} rows into nutrient_info")


import numpy as np

print("Rows with inf:", np.isinf(df_metrics.select_dtypes(include=[float])).sum())

df_metrics_clean = df_metrics.where(pd.notnull(df_metrics), None)

import numpy as np

# Replace numpy NaN, 'nan' strings, and inf with None
df_metrics_clean = df_metrics.replace({
    np.nan: None,
    'nan': None,
    np.inf: None,
    -np.inf: None
})


data_metrics = df_metrics_clean[[
    'code', 'sugar_to_carb_ratio', 'calorie_category',
    'sugar_category', 'is_ultra_processed'
]].values.tolist()

mycursor.executemany("""
    INSERT IGNORE INTO chococrunch_analysis.derived_metrics (
        code, sugar_to_carb_ratio, calorie_category,
        sugar_category, is_ultra_processed
    ) VALUES (%s, %s, %s, %s, %s)
""", data_metrics)

connection.commit()
print(f"Bulk inserted {len(data_metrics)} rows into derived_metrics")



