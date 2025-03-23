import pandas as pd
import numpy as np
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import psutil

def log_memory_usage(prefix=""):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB
    print(f"{prefix} Memory usage: {mem_mb:.2f} MB")

nltk_data_path = "/app/nltk_data"

# Ensure the directory exists
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

# Add to NLTK's path
nltk.data.path.append(nltk_data_path)

# ✅ Only download if the data doesn't already exist
stopwords_path = os.path.join(nltk_data_path, "corpora/stopwords.zip")
wordnet_path = os.path.join(nltk_data_path, "corpora/wordnet.zip")

if not os.path.exists(stopwords_path):
    nltk.download("stopwords", download_dir=nltk_data_path)

if not os.path.exists(wordnet_path):
    nltk.download("wordnet", download_dir=nltk_data_path)

lemmatizer = WordNetLemmatizer()
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load datasets
df_ibm = pd.read_csv("https://raw.githubusercontent.com/WinNatch/IBM_SKILL/main/IBM%20Course%20Data%20_%20Rating%20-%20Data.csv", encoding="ISO-8859-1")
df_jobs = pd.read_csv("https://raw.githubusercontent.com/WinNatch/IBM_SKILL/main/Restructured_Job_Skills_Data.csv")
df_university = pd.read_json("https://raw.githubusercontent.com/WinNatch/IBM_SKILL/main/UoB_Business_School_Data.json", lines=True)

# IBM Data Cleaning
df_ibm = df_ibm[df_ibm["Level"].str.lower() != "application"]
df_ibm = df_ibm.rename(columns={"Desciption": "Description"})

# Select relevant columns
df_ibm = df_ibm[["Course_Name", "Description", "ILO", "Tags", "Duration", "Rating", "Level", "URL", "5&4 Stars Percentage"]]
df_jobs = df_jobs[["Job Role", "Skill", "Percentage"]]
df_university = df_university[["Programme", "Overview", "Programme_Structure", "Career_Prospects", "Programme Catalogue", "Programme Details Mixed"]]

# Convert rating to numeric and drop NaN values
df_ibm["Rating"] = pd.to_numeric(df_ibm["Rating"], errors='coerce')
df_ibm = df_ibm.dropna(subset=["Rating"])

# Custom stopwords
custom_stopwords = set(stopwords.words("english")).union({
    "in", "on", "at", "by", "for", "with", "about", "as", "into", "through", "between",
    "and", "or", "but", "because", "so", "although", "unit",
    "like", "just", "very", "also", "more", "way", "course", "know", "youll", "team", "field", "learn", "skill",
    "learning", "step", "help", "cause", "human", "follow", "car", "type", "difference", "world", "discover",
    "design", "tool", "describe", "differentiate", "define", "explain", "learns", "result", "rule", "need",
    "value", "use", "identify", "explore", "create", "role", "function", "compare", "recognize"
})

# Define software & technique keywords
software_keywords = {kw.lower() for kw in {
    "Python", "Tableau", "Excel", "Power BI", "SQL", "Structured Query Language",
    "TensorFlow", "PyTorch", "AWS", "Google Cloud", "Azure", "Hadoop", "Spark",
    "Kubernetes", "Docker", "Jupyter", "PostgreSQL", "MongoDB", "MySQL", "BigQuery",
    "Snowflake", "SAS", "STATA", "MATLAB", "Scikit-learn", "Pandas", "NumPy",
    "Seaborn", "ggplot2", "Django", "Flask", "FastAPI", "IBM Watsonx", "IBM",
    "CSS", "HTML", "JavaScript", "Watson", "Studio", "Operations",
    "Agile", "UX", "Review", "R"
}}

technique_keywords = {kw.lower() for kw in {
    "NLP", "Natural Language Processing", "CNN", "Convolutional Neural Network",
    "RNN", "Recurrent Neural Network", "LSTM", "Long Short-Term Memory",
    "Generative Adversarial Network", "XGBoost", "Reinforcement Learning",
    "Gradient Boosting", "SVM", "Support Vector Machine", "Decision Tree",
    "Random Forest", "Clustering", "K-Means", "PCA", "Principal Component Analysis",
    "Feature Engineering", "Data Augmentation", "Machine Learning", "Deep Learning",
    "Neural Networks", "Cybersecurity", "Threats", "Sustainability", "Ethics", "Regression", "chatbot"
}}

def process_user_response(user_response):
    log_memory_usage("Start processing: ")
    
    # Use user_response instead of manually entering input
    user_input_combined = user_response

    # Declare df_ibm, df_university, and df_jobs as global
    global df_ibm, df_university, df_jobs

    # Preprocessing function
    def preprocess_text(text):
        if pd.isna(text):
            return ""
        text = re.sub(r'[^\w\s]', '', text).lower()
        tokens = text.split()
        tokens = [word for word in tokens if word not in custom_stopwords]
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return " ".join(tokens)

    # Preprocess data
    df_ibm["Processed_Text"] = df_ibm[["Description", "ILO", "Tags"]].astype(str).apply(lambda x: " ".join(x), axis=1).apply(preprocess_text)
    df_university["Processed_Text"] = df_university[["Overview", "Programme_Structure", "Career_Prospects", "Programme Catalogue", "Programme Details Mixed"]].astype(str).apply(lambda x: " ".join(x), axis=1).apply(preprocess_text)
    df_jobs["Processed_Text"] = df_jobs[["Job Role", "Skill"]].astype(str).apply(lambda x: " ".join(x), axis=1).apply(preprocess_text)

    # Compute job role similarity using TF-IDF first
    vectorizer_jobs = TfidfVectorizer()
    tfidf_jobs = vectorizer_jobs.fit_transform(df_jobs["Job Role"])
    tfidf_user_job = vectorizer_jobs.transform([preprocess_text(user_input_combined)])
    similarity_jobs_tfidf = cosine_similarity(tfidf_user_job, tfidf_jobs).flatten()

    # Compute job role similarity using BERT
    user_embedding = bert_model.encode(user_input_combined, convert_to_tensor=True)
    job_embeddings = bert_model.encode(df_jobs["Job Role"].tolist(), convert_to_tensor=True)
    job_similarities = util.pytorch_cos_sim(user_embedding, job_embeddings).flatten().cpu().numpy()
    job_similarities_bert = util.pytorch_cos_sim(user_embedding, job_embeddings).flatten().cpu().numpy()

    # Combine both similarity measures
    final_job_scores = (similarity_jobs_tfidf * 0.4) + (job_similarities_bert * 0.6)
    best_job_index = np.argmax(final_job_scores)
    best_job_score = final_job_scores[best_job_index]
    best_matching_job = df_jobs.iloc[best_job_index]["Job Role"]
    best_matching_job_skills = df_jobs[df_jobs["Job Role"] == best_matching_job]["Skill"].tolist()

    # Compute TF-IDF similarity for university programs
    vectorizer = TfidfVectorizer()
    combined_corpus = list(df_university["Processed_Text"]) + [preprocess_text(user_input_combined)]
    tfidf_university = vectorizer.fit_transform(combined_corpus[:-1])
    tfidf_user = vectorizer.transform([combined_corpus[-1]])
    similarity_university = cosine_similarity(tfidf_user, tfidf_university).flatten()

    best_university_index = np.argmax(similarity_university)
    best_university_score = similarity_university[best_university_index]
    best_matching_university = df_university.iloc[best_university_index]["Programme"]

    # Compute TF-IDF similarity for IBM courses
    tfidf_ibm = vectorizer.fit_transform(df_ibm["Processed_Text"])
    tfidf_user_ibm = vectorizer.transform([preprocess_text(user_input_combined)])
    similarity_ibm = cosine_similarity(tfidf_user_ibm, tfidf_ibm).flatten()

    df_ibm["Relevance_Score"] = similarity_ibm
    df_ibm = df_ibm[df_ibm["Rating"] >= 4.0]

    # **Increase Weight for Technical Keywords**
    boost_factor = 1.5  # Increased to give higher weight to tech skills
    def boost_matching_keywords(text, base_score):
        tokens = text.lower().split()
        keyword_count = sum(1 for word in tokens if word in software_keywords or word in technique_keywords)
        return base_score * (1 + (boost_factor * keyword_count))  # Multiply instead of add

    df_ibm["Relevance_Score"] = df_ibm.apply(lambda row: boost_matching_keywords(row["Processed_Text"], row["Relevance_Score"]), axis=1)

    # Dynamic weight adjustment based on similarity scores
    weights = {'ibm_course': 0.4, 'job_role': 0.3, 'university_program': 0.3}

    if best_job_score < 0.4:
        weights['ibm_course'] += weights['job_role'] * 0.5
        weights['university_program'] += weights['job_role'] * 0.5
        weights['job_role'] = 0.0

    # Normalize weights
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}

    # Compute final relevance score
    df_ibm["Final_Relevance_Score"] = (
        df_ibm["Relevance_Score"] * weights['ibm_course'] +
        best_job_score * weights['job_role'] +
        best_university_score * weights['university_program'])

    # Normalize scores to 0-100% scale
    df_ibm["Matching_Percentage"] = (df_ibm["Final_Relevance_Score"] / df_ibm["Final_Relevance_Score"].max()) * 100
    df_ibm["Matching_Percentage"] = df_ibm["Matching_Percentage"].apply(lambda x: min(100, max(10, x)))

    # Ensure percentages look reasonable
    df_ibm["Matching_Percentage"] = df_ibm["Matching_Percentage"].apply(lambda x: min(100, max(10, x)))

    # Sort and get top recommendations
    top_match_course = df_ibm.sort_values(by="Final_Relevance_Score", ascending=False).head(3)

    # Create the recommendation dictionary correctly
    recommendation = {
        "top_ibm_courses": []
    }

    # Iterate correctly
    for _, row in top_match_course.iterrows():
        # Try extracting the percentage value and converting it to an integer
        try:
            high_rated_percentage = int(float(row["5&4 Stars Percentage"]))  # Convert to int if possible
            high_rated_percentage = f"{high_rated_percentage}%"
        except (ValueError, KeyError, TypeError):
            high_rated_percentage = "N/A"  # If conversion fails, use "N/A"

        recommendation["top_ibm_courses"].append({
            "Course_Name": row["Course_Name"],
            "Course_URL": row.get("URL", "No URL available"),
            "Rating": row.get("Rating", "N/A"),
            "High_Rated_Percentage": high_rated_percentage
        })

    return recommendation