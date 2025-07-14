import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text) 
    text = re.sub(r"\S+@\S+", "", text)         
    text = re.sub(r"\+?\d[\d\s\-()]{7,}", "", text) 
    text = re.sub(r"[^a-z\s]", " ", text)       
    text = re.sub(r"\s+", " ", text).strip()    

    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return " ".join(tokens)

def merge_resume_fields(row):
    fields = [
        'career_objective',
        'skills',
        'degree_names',
        'major_field_of_studies',
        'educational_institution_name',
        'professional_company_names',
        'positions',
        'responsibilities',
        'extra_curricular_activity_types',
        'extra_curricular_organization_names',
        'languages',
        'certification_skills'
    ]
    return " ".join([str(row[col]) for col in fields if pd.notnull(row[col])])

def merge_job_fields(row):
    return f"{row['Job Title']} {row['Job Description']}"

def preprocess_dataframe(df, text_column):
    df[text_column] = df[text_column].apply(clean_text)
    return df

def main():
    resumes = pd.read_csv('../data/resume_data.csv')
    jobs = pd.read_csv('../data/job_des.csv')

    # Merging and cleaning resume data

    resumes['resume_text'] = resumes.apply(merge_resume_fields,axis=1)
    resumes = preprocess_dataframe(resumes,"resume_text")
    resumes[['resume_text']].to_csv('../data/resume_clean.csv',index=False)

    # Merging and cleaning job data

    jobs['job_text'] = jobs.apply(merge_job_fields, axis=1)
    jobs = preprocess_dataframe(jobs, 'job_text')
    jobs[['job_text']].to_csv('../data/jobs_clean.csv', index=False)


if __name__ == "__main__":
    main()


