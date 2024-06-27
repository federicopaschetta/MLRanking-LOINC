# ML Ranking (LOINC)

## Overview
This project implements a Machine Learning Ranking system for LOINC (Logical Observation Identifiers Names and Codes) data. It uses a pointwise approach to rank medical laboratory observations based on their relevance to specific queries.

## Motivation
In health data analysis, interpreting lab results efficiently is crucial. This project aims to apply machine learning techniques to rank and retrieve relevant LOINC codes based on textual queries, facilitating better understanding and analysis of medical data.

## Features
- Pointwise ML Ranking approach
- TF-IDF (Term Frequency-Inverse Document Frequency) based scoring
- Cosine similarity for relevance calculation
- Binary relevance labeling
- Dataset expansion capabilities

## Technologies Used
- Python
- Pandas
- Scikit-learn (TfidfVectorizer, cosine_similarity)
- Jupyter Notebook

## Datasets
1. loinc_dataset-v2.xlsx: Initial dataset with 70 rows per query
2. LoincTableCore.csv: Extended LOINC dataset with over 100,000 entries
3. loinc_property.csv: LOINC property attribute mappings
4. loinc_system.csv: LOINC system attribute mappings

## Implementation Details
1. Data Preparation:
   - Import and process Excel and CSV files
   - Map property and system data to meaningful labels

2. Ranking Calculation:
   - Remove stopwords from queries
   - Generate TF-IDF scores for documents
   - Calculate cosine similarity between query and document vectors

3. Evaluation:
   - Compare automated rankings with manual binary relevance labels
   - Generate confusion matrices for each query

4. Dataset Expansion:
   - Apply ranking methodology to the full LOINC dataset

## Results
The model shows good performance in ranking based on term occurrences but faces challenges with synonyms and technical terms. Performance varies by query:

1. "glucose in blood": Perfect precision and recall
2. "bilirubin in plasma": 100% precision, 86% recall
3. "white blood cells count": 78% precision, 100% recall

## Project Structure
```
|-- LoincTableCore.csv
|-- loinc_dataset-v2.xlsx
|-- loinc_property.csv
|-- loinc_system.csv
|-- mlranking.ipynb
|-- mlranking.py
|-- training/
    |-- bilirubin_in_plasma.xlsx
    |-- breast_cancer.xlsx
    |-- glucose_in_blood.xlsx
    |-- white_blood_cells_count.xlsx
    |-- init_bilirubin_in_plasma.xlsx
    |-- init_glucose_in_blood.xlsx
    |-- init_white_blood_cells_count.xlsx
```

## How to Run
1. Ensure you have Python installed with the required libraries (Pandas, Scikit-learn)
2. Place all dataset files in the project root directory
3. Run the Jupyter notebook `mlranking.ipynb` or the Python script `mlranking.py`

## Future Improvements
- Incorporate Natural Language Processing modules to handle synonyms and technical terms better
- Explore other machine learning models for ranking
- Implement cross-validation for more robust evaluation

## Contributors
- Federico Paschetta
- Cecilia Peccolo
- Nicola Maria D'Angelo

## Acknowledgments
- Universidad Politécnica de Madrid
- LOINC for providing the datasets