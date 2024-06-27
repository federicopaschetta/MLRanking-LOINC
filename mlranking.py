# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ### Importing necessary libraries

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix, recall_score, precision_score
import re

# ### Reading datasets given

old_glucose_df = pd.read_excel('loinc_dataset-v2.xlsx', header=2, sheet_name='glucose in blood')
old_glucose_df.head()

old_bilirubin_df = pd.read_excel('loinc_dataset-v2.xlsx', header=2, sheet_name='bilirubin in plasma')
old_bilirubin_df.head()

old_white_cells_df = pd.read_excel('loinc_dataset-v2.xlsx', header=2, sheet_name='White blood cells count')
old_white_cells_df.head()

# ### Reading dataset with all LOINC codes

new_df = pd.read_csv('LoincTableCore.csv')
new_df.head()

# ### Reducing dataset keeping only attributes in first dataset

df_extended = new_df[[col for col in new_df.columns if col.lower() in list(old_glucose_df.columns)]]
for col in df_extended.columns:
    df_extended.rename(columns={col: col.lower()}, inplace=True)
df_extended.head()


# ### Mapping properties abbreviations to meaning

def read_property_dict(filepath: str) -> dict:
    prop_dict = {}
    df = pd.read_csv(filepath, header=0)
    for index, row in df.iterrows():
        prop_dict[row.iloc[1]] = row.iloc[2]
    return prop_dict



# ### Calling property map function

prop_filepath = 'loinc_property.csv'
property_dict = read_property_dict(prop_filepath)
print(property_dict)


# ### Mapping system abbreviations to meaning

def read_system_dict(filepath: str) -> dict:
    syst_dict = {}
    df = pd.read_csv(filepath, header=0)
    for index, row in df.iterrows():
        syst_dict[row.iloc[0]] = row.iloc[1]
    return syst_dict


# ### Calling system map function

syst_filepath = 'loinc_system.csv'
system_dict = read_system_dict(syst_filepath)
print(system_dict)


# ### Updating dataset with meanings instead of abbreviations

def update_df(df, prop_dict, system_dict):
    df_copy = df.copy()
    for index, row in df.iterrows():
        if row['property'] in prop_dict:
            df_copy.at[index, 'property'] = prop_dict[row['property']]
        if row['system'] in system_dict:
            df_copy.at[index, 'system'] = system_dict[row['system']]
    return df_copy



old_bilirubin_df = update_df(old_bilirubin_df, property_dict, system_dict)
old_glucose_df = update_df(old_glucose_df, property_dict, system_dict)
old_white_cells_df = update_df(old_white_cells_df, property_dict, system_dict)

mapped_df = update_df(df_extended, property_dict, system_dict)
mapped_df


# ### Computing relevance score with TF-IDF

def calculate_ranking(df, query: str):
    query = re.sub(r'\bin\b', '', query).strip()
    vocabulary = query.split()
    last_df = df.copy()
    if 'relevance' in df.columns:
        df = df.drop(columns=['relevance'])
    combined_text = [' '.join(row) for row in zip(df['long_common_name'], df['component'])]
    vectorizer = TfidfVectorizer(vocabulary=vocabulary)
    X = vectorizer.fit_transform(combined_text)
    query_vector = vectorizer.transform([query])
    similarity_scores = cosine_similarity(X, query_vector).flatten()
    last_df['score'] = similarity_scores
    return last_df


def evaluation(df, query, threshold):
    new = calculate_ranking(df, query)
    y_pred = (new['score'] >= threshold).astype(int)
    cm = confusion_matrix(new['relevance'], y_pred)
    precision = precision_score(new['relevance'], y_pred)
    recall = recall_score(new['relevance'], y_pred)
    print("Results for the query:", query)
    print("\nConfusion Matrix:")
    print(cm)
    print("\nPrecision:", precision)
    print("Recall:", recall)


# ### Checking TF-IDF reliability on provided training dataset

evaluation(old_bilirubin_df, 'bilirubin in plasma', 0.5)

old_bilirubin_df = calculate_ranking(old_bilirubin_df, "bilirubin in plasma")
old_bilirubin_df = old_bilirubin_df.sort_values(by='score', ascending=False)
old_bilirubin_df.to_excel('./training/init_bilirubin_in_plasma.xlsx', index=False)

evaluation(old_glucose_df, 'glucose in blood', 0.1)

old_glucose_df = calculate_ranking(old_glucose_df, "glucose in blood")
old_glucose_df = old_glucose_df.sort_values(by='score', ascending=False)
old_glucose_df.to_excel('./training/init_glucose_in_blood.xlsx', index=False)

evaluation(old_white_cells_df, 'White blood cells count', 0.1)

old_white_cells_df = calculate_ranking(old_white_cells_df, "white blood cells count")
old_white_cells_df = old_white_cells_df.sort_values(by='score', ascending=False)
old_white_cells_df.to_excel('./training/init_white_blood_cells_count.xlsx', index=False)

# ### Extending dataset with all LOINC data relevant for each query

bilirubin_df = calculate_ranking(mapped_df, "bilirubin in plasma")
bilirubin_df = bilirubin_df.sort_values(by='score', ascending=False)
bilirubin_df[bilirubin_df['score']>0].to_excel('./training/bilirubin_in_plasma.xlsx', index=False)

glucose_in_blood_df = calculate_ranking(mapped_df, "glucose in blood")
glucose_in_blood_df = glucose_in_blood_df.sort_values(by='score', ascending=False)
glucose_in_blood_df[glucose_in_blood_df['score']>0].to_excel('./training/glucose_in_blood.xlsx', index=False)

white_blood_cells_count_df = calculate_ranking(mapped_df, "white blood cells count")
white_blood_cells_count_df = white_blood_cells_count_df.sort_values(by='score', ascending=False)
white_blood_cells_count_df[white_blood_cells_count_df['score']>0].to_excel('./training/white_blood_cells_count.xlsx', index=False)

breast_cancer_df = calculate_ranking(mapped_df, "breast cancer")
breast_cancer_df = breast_cancer_df.sort_values(by='score', ascending=False)
breast_cancer_df[breast_cancer_df['score']>0].to_excel('./training/breast_cancer.xlsx', index=False)
