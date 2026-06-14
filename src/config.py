"""
config.py — JD-derived requirements, skill taxonomies, scoring weights.

All domain knowledge about the Senior AI Engineer role is encoded here so that
scoring modules can stay logic-only and this file is the single source of truth
for "what the JD actually means."
"""

# ─────────────────────────────────────────────────────────────────────────────
# JD metadata
# ─────────────────────────────────────────────────────────────────────────────
JD_TITLE = "Senior AI Engineer — Founding Team"
JD_COMPANY = "Redrob AI"
JD_LOCATION_PREFERRED = ["Pune", "Noida"]
JD_LOCATION_ACCEPTABLE = [
    "Hyderabad", "Mumbai", "Delhi NCR", "Gurgaon", "Gurugram",
    "Bangalore", "Bengaluru", "Chennai", "Kolkata",
]
JD_COUNTRY = "India"
JD_EXPERIENCE_SWEET = (5, 9)       # ideal band
JD_EXPERIENCE_ACCEPTABLE = (3, 15) # acceptable with penalty
JD_WORK_MODE = "hybrid"
JD_NOTICE_IDEAL_DAYS = 30

# ─────────────────────────────────────────────────────────────────────────────
# Scoring weights (must sum to 1.0)
# ─────────────────────────────────────────────────────────────────────────────
WEIGHTS = {
    "semantic":           0.15,   # TF-IDF / embedding cosine similarity
    "title_role":         0.22,   # current & historical title relevance
    "skill_match":        0.18,   # core & nice-to-have skill coverage
    "skill_credibility":  0.08,   # cross-validation of skills vs career
    "career_trajectory":  0.15,   # product-co experience, ML systems shipped
    "experience_band":    0.08,   # proximity to 5-9 yr sweet spot
    "location":           0.05,   # India / preferred city
    "education":          0.03,   # CS/ML field, institution tier
    "certifications":     0.03,   # relevant certifications & GitHub
    "behavioral":         0.03,   # small additive component (rest applied as multiplier)
}

# ─────────────────────────────────────────────────────────────────────────────
# Skill taxonomy — mapped from JD sections
# ─────────────────────────────────────────────────────────────────────────────

# "Things you absolutely need" — highest weight
CORE_SKILLS_TIER1 = {
    # Embeddings-based retrieval
    "sentence-transformers", "sentence transformers", "openai embeddings",
    "bge", "e5", "embeddings", "sentence embeddings", "text embeddings",
    "semantic search", "dense retrieval", "embedding",

    # Vector databases / hybrid search
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
    "elasticsearch", "faiss", "vector database", "vector search",
    "hybrid search", "annoy", "chromadb", "chroma",

    # NLP / IR fundamentals
    "nlp", "natural language processing", "information retrieval",
    "text classification", "ner", "named entity recognition",
    "text mining", "text analytics", "spacy", "nltk",

    # Ranking / Search / Recommendation
    "ranking", "search", "recommendation", "retrieval",
    "learning to rank", "search engine", "recommendation system",
    "collaborative filtering", "content-based filtering",

    # Evaluation
    "ndcg", "mrr", "map", "a/b testing", "evaluation",
    "precision", "recall", "f1",

    # Strong Python & ML fundamentals
    "python", "pytorch", "tensorflow", "keras",
    "machine learning", "deep learning", "neural networks",
    "transformers", "hugging face", "huggingface",
    "scikit-learn", "sklearn",
}

# "Things we'd like you to have" — moderate weight
CORE_SKILLS_TIER2 = {
    "lora", "qlora", "peft", "fine-tuning", "fine-tuning llms",
    "model fine-tuning", "finetuning",
    "xgboost", "lightgbm", "gradient boosting", "catboost",
    "distributed systems", "kubernetes", "docker",
    "mlops", "mlflow", "kubeflow", "weights & biases", "wandb",
    "rag", "retrieval augmented generation",
    "langchain", "llamaindex", "llama index",
    "prompt engineering", "llm", "large language model",
    "gpt", "bert", "t5", "llama",
    "data engineering", "spark", "airflow", "kafka",
    "feature engineering", "feature store",
    "model serving", "triton", "bentoml", "seldon",
    "statistical modeling", "bayesian",
    "gans", "generative ai", "diffusion models",
    "reinforcement learning",
    "object detection", "image classification", "computer vision",
    "speech recognition", "tts", "asr",
}

# Explicitly non-relevant skills (noise indicators when present without core)
NON_RELEVANT_SKILLS = {
    "photoshop", "illustrator", "canva", "figma",
    "excel", "powerpoint", "word",
    "seo", "content writing", "copywriting",
    "accounting", "tally", "sap",
    "six sigma", "lean", "pmp",
    "autocad", "solidworks", "creo", "ansys", "catia",
    "marketing", "sales", "crm", "salesforce",
    "react", "angular", "vue", "next.js",
    "tailwind", "css", "html", "bootstrap",
    "redux", "webpack", "node.js",
    "javascript", "typescript",
    "java", "c#", ".net", "spring boot",
    "ruby", "php", "wordpress",
}

# ─────────────────────────────────────────────────────────────────────────────
# Title taxonomy
# ─────────────────────────────────────────────────────────────────────────────

# Titles that are strong positive signals for this JD
AI_ML_TITLES = [
    "ai engineer", "ml engineer", "machine learning engineer",
    "data scientist", "research engineer", "nlp engineer",
    "deep learning engineer", "applied scientist",
    "senior ai engineer", "senior ml engineer",
    "senior machine learning engineer", "junior ml engineer",
    "senior data scientist", "lead data scientist",
    "ai/ml engineer", "ai researcher", "ml researcher",
    "applied machine learning", "staff ml engineer",
    "principal ml engineer", "senior research engineer",
    "machine learning scientist", "senior nlp engineer",
]

# Titles that are moderately relevant (could do ML work)
ADJACENT_TITLES = [
    "backend engineer", "software engineer", "data engineer",
    "full stack engineer", "platform engineer", "senior software engineer",
    "staff engineer", "senior data engineer", "analytics engineer",
    "devops engineer", "infrastructure engineer", "senior backend engineer",
]

# Titles that are strong negative signals — the "keyword stuffer" traps
NON_RELEVANT_TITLES = [
    "marketing manager", "hr manager", "accountant",
    "civil engineer", "mechanical engineer", "customer support",
    "operations manager", "sales executive", "content writer",
    "graphic designer", "project manager", "business analyst",
    "financial analyst", "admin", "receptionist",
    "supply chain", "logistics", "procurement",
    "teacher", "professor", "lecturer",
]

# ─────────────────────────────────────────────────────────────────────────────
# Company classification
# ─────────────────────────────────────────────────────────────────────────────

# Pure consulting/services firms — JD explicitly disqualifies candidates
# whose ENTIRE career is at these companies
CONSULTING_SERVICES_FIRMS = {
    "tcs", "tata consultancy services",
    "infosys",
    "wipro",
    "accenture",
    "cognizant",
    "capgemini",
    "hcl", "hcl technologies",
    "tech mahindra",
    "mindtree", "ltimindtree",
    "mphasis",
    "l&t infotech", "lti",
    "hexaware",
    "coforge", "niit technologies",
    "persistent systems",
    "zensar",
    "cyient",
    "birlasoft",
    "sonata software",
    "mastech",
    "ust global", "ust",
}

# ─────────────────────────────────────────────────────────────────────────────
# Career description keywords for ML systems evidence
# ─────────────────────────────────────────────────────────────────────────────

# Keywords in career descriptions that indicate real AI/ML systems work
ML_SYSTEMS_KEYWORDS = [
    "machine learning", "deep learning", "neural network",
    "embedding", "vector", "retrieval", "ranking", "recommendation",
    "model training", "model serving", "inference",
    "pytorch", "tensorflow", "transformer",
    "fine-tun", "llm", "language model", "bert", "gpt",
    "nlp", "natural language", "text classification",
    "search engine", "search system", "information retrieval",
    "feature engineering", "feature pipeline", "feature store",
    "a/b test", "evaluation", "metrics",
    "ndcg", "precision", "recall", "auc",
    "data pipeline", "ml pipeline", "training pipeline",
    "mlops", "model monitor", "model deploy",
    "recommendation system", "recommender", "collaborative filtering",
    "semantic search", "vector database", "similarity",
    "rag", "retrieval augmented",
    "production model", "real-time inference", "batch inference",
    "data science", "statistical model", "predictive model",
]

# Keywords indicating product/systems work (not just consulting)
PRODUCT_SYSTEMS_KEYWORDS = [
    "shipped", "deployed", "production", "real user",
    "built", "designed", "architected", "implemented",
    "scale", "latency", "throughput", "uptime",
    "microservice", "api", "backend", "infrastructure",
    "real-time", "batch processing", "stream processing",
    "product", "platform", "saas",
]

# ─────────────────────────────────────────────────────────────────────────────
# JD text for semantic matching (condensed)
# ─────────────────────────────────────────────────────────────────────────────
JD_SEMANTIC_TEXT = """
Senior AI Engineer for an AI-native talent intelligence platform.
Own the intelligence layer: ranking, retrieval, and matching systems for
recruiter candidate search and candidate role search.
Build embeddings-based retrieval, hybrid search with vector databases,
LLM-based re-ranking. Set up evaluation infrastructure with offline benchmarks
and online A/B testing, recruiter-feedback loops.
Must have production experience with embeddings retrieval systems like
sentence-transformers, BGE, E5 deployed to real users. Experience with
vector databases like Pinecone, Weaviate, Qdrant, Milvus, FAISS, Elasticsearch.
Strong Python. Evaluation frameworks for ranking systems: NDCG, MRR, MAP.
Nice to have: LLM fine-tuning with LoRA QLoRA PEFT, learning-to-rank with
XGBoost, HR-tech recruiting marketplace experience, distributed systems,
open-source contributions in AI ML space.
Shipped end-to-end ranking search recommendation system to real users.
Machine learning deep learning NLP information retrieval natural language
processing text classification semantic search neural networks transformers
PyTorch TensorFlow model serving inference pipeline data engineering.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Behavioral signal thresholds
# ─────────────────────────────────────────────────────────────────────────────
BEHAVIORAL_THRESHOLDS = {
    "response_rate_good": 0.50,
    "response_rate_bad": 0.15,
    "response_time_good_hrs": 48,
    "response_time_bad_hrs": 120,
    "profile_completeness_good": 70,
    "profile_completeness_bad": 30,
    "github_activity_good": 40,
    "notice_period_ideal": 30,
    "notice_period_max": 90,
    "inactive_days_warning": 90,
    "inactive_days_critical": 180,
    "interview_completion_good": 0.70,
    "interview_completion_bad": 0.30,
}
