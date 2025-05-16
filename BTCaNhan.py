import streamlit as st
import string
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

import spacy
import fasttext.util
from gensim.downloader import load as gensim_load

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import unicodedata
from sklearn.pipeline import make_pipeline

nltk.download('stopwords')
nltk.download('wordnet')
nlp = spacy.load("en_core_web_sm")

st.set_page_config(page_title="NLP Processing App", layout="centered")
st.markdown("""
    <style>
        .main {background-color: #f9f9f9;}
        .block-container {
            max-width: 900px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3 {
            color: #274060;
            font-family: 'Segoe UI', sans-serif;
        }
        .stTextInput > div > input {
            background-color: #ffffff;
            color: black;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Ứng dụng Xử Lý Ngôn Ngữ Tự Nhiên")
st.caption("Bao gồm các bước: Tăng cường dữ liệu, Tiền xử lý, Vector hóa & Nhúng văn bản")

# -----------------------------
# Nhập dữ liệu
# -----------------------------
st.header("Nhập dữ liệu")
input_method = st.selectbox("Chọn nguồn dữ liệu", ["Nhập tay", "Tải dataset CSV", "Cào từ web"])
data = []

if input_method == "Nhập tay":
    user_input = st.text_area("Nhập văn bản", height=150)
    if user_input:
        data = [user_input]

elif input_method == "Tải dataset CSV":
    uploaded_file = st.file_uploader("Tải lên file CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        cols = st.multiselect("Chọn 1 hoặc nhiều cột chứa văn bản", df.columns)
        if cols:
            df["__combined__"] = df[cols].astype(str).agg(" ".join, axis=1)
            data = df["__combined__"].dropna().tolist()
            st.success(f"Đã kết hợp {len(cols)} cột thành 1 văn bản.")
            st.subheader("Xem trước 10 dòng dữ liệu sau khi kết hợp:")
            st.dataframe(df[cols].head(10))

elif input_method == "Cào từ web":
    url = st.text_input("🌐 Nhập URL cần cào dữ liệu:")
    if st.button("🔍 Cào nội dung") and url:
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all("p")
            data = [p.get_text() for p in paragraphs if len(p.get_text()) > 30]
            st.success(f"Đã cào được {len(data)} đoạn văn.")
            st.write(data[:10])
        except:
            st.error("Không thể cào dữ liệu từ URL.")

# -----------------------------
# Bước 1: Tăng cường dữ liệu
# -----------------------------
st.header("Bước 1: Tăng cường dữ liệu")
option_aug = st.selectbox("Chọn phương pháp tăng cường", [
    "Không thay đổi", "Đảo ngược", "Thay từ đồng nghĩa", "Thay thế thực thể", "Thêm nhiễu"])
aug_param = st.text_input("Nhập tham số (nếu có):", help="Ví dụ: từ_cũ từ_mới hoặc thực_thể_cũ,thực_thể_mới")

def apply_augmentation(text):
    if option_aug == "Đảo ngược":
        return text[::-1]
    elif option_aug == "Thay từ đồng nghĩa":
        try:
            old, new = aug_param.split()
            return text.replace(old, new)
        except:
            return text
    elif option_aug == "Thay thế thực thể":
        try:
            old, new = aug_param.split(",")
            return text.replace(old.strip(), new.strip())
        except:
            return text
    elif option_aug == "Thêm nhiễu":
        return "".join([char + "*" if char.isalpha() else char for char in text])
    return text

if data:
    augmented_data = [apply_augmentation(t) for t in data]
    st.subheader("Kết quả tăng cường (10 dòng đầu):")
    st.write(augmented_data[:10])
else:
    augmented_data = []
    st.warning("⚠️ Chưa có dữ liệu để tăng cường.")

# -----------------------------
# Bước 2: Tiền xử lý văn bản
# -----------------------------
st.header("Bước 2: Tiền xử lý văn bản")
step2_ops = st.multiselect("Chọn các thao tác tiền xử lý:", [
    "Tách từ (Split)", "Tokenize (Spacy)", "Loại bỏ Stopwords", "Loại bỏ Dấu câu",
    "Stemming", "Lemmatization", "POS Tagging"])

ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    try:
        if "Loại bỏ Dấu câu" in step2_ops:
            text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = text.split()
        if "Loại bỏ Stopwords" in step2_ops:
            stop_words = set(stopwords.words("english"))
            tokens = [t for t in tokens if t.lower() not in stop_words]
        if "Stemming" in step2_ops:
            tokens = [ps.stem(t) for t in tokens]
        if "Lemmatization" in step2_ops:
            tokens = [lemmatizer.lemmatize(t) for t in tokens]
        if "POS Tagging" in step2_ops:
            doc = nlp(" ".join(tokens))
            tokens = [f"{t.text}/{t.pos_}" for t in doc]
        if "Tokenize (Spacy)" in step2_ops:
            doc = nlp(text)
            tokens = [f"{t.text}/{t.pos_}" for t in doc]
        if "Tách từ (Split)" in step2_ops:
            tokens = [f'"{t}"' for t in tokens]
        return " ".join(tokens)
    except Exception as e:
        return f"[Lỗi xử lý: {e}]"

if augmented_data:
    processed_data = [preprocess(t) for t in augmented_data]
    st.subheader("Kết quả sau tiền xử lý (10 dòng đầu):")
    st.write(processed_data[:10])
else:
    st.warning("⚠️ Chưa có dữ liệu để xử lý.")

# --- Bước 3: Vector hóa ---
st.header("Bước 3: Vector hóa dữ liệu")
vec_option = st.selectbox("Chọn phương pháp vector hóa", ["One-Hot Encoding", "Bag of Words"])

def one_hot_encoding(text):
    words = np.array(text.replace('"', '').split()).reshape(-1, 1)
    encoder = OneHotEncoder(sparse_output=False)
    encoded = encoder.fit_transform(words)
    return encoded

def bag_of_words(text):
    vectorizer = CountVectorizer()
    bag = vectorizer.fit_transform([text.replace('"', '')]).toarray()
    return bag, vectorizer.get_feature_names_out()

if processed_data:
    text = processed_data[0]  # lấy dòng đầu tiên để xử lý
    if vec_option == "One-Hot Encoding":
        encoded = one_hot_encoding(text)
        st.write("One-Hot Encoding Vector:")
        st.write(encoded)
    elif vec_option == "Bag of Words":
        bag, feats = bag_of_words(text)
        st.write("Bag of Words Vector:")
        st.write(f"Features: {feats}")
        st.write(bag)

else:
    st.warning("⚠️ Vui lòng nhập văn bản để vector hóa.")

# --- Bước 4: Nhúng văn bản ---
st.header("Bước 4: Nhúng văn bản")

embed_option = st.selectbox(
    "Chọn phương pháp nhúng", 
    ["Bag of N-grams", "TF-IDF", "FastText", "GloVe", "BERT", "GPT-2"]
)

# FastText
@st.cache_data
def load_fasttext():
    fasttext.util.download_model('en', if_exists='ignore')
    return fasttext.load_model('cc.en.300.bin')

# GloVe
@st.cache_data
def load_glove():
    return gensim_load("glove-wiki-gigaword-50")

# BERT
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2Model
import torch

@st.cache_resource
def load_bert():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    return tokenizer, model

# GPT-2
@st.cache_resource
def load_gpt2():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2")
    return tokenizer, model

# ===== Các hàm xử lý nhúng =====

def bag_of_ngrams(text, n=2):
    vectorizer = CountVectorizer(ngram_range=(n, n))
    bag = vectorizer.fit_transform([text.replace('"', '')]).toarray()
    return bag, vectorizer.get_feature_names_out()

def tfidf_vectorization(text):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([text.replace('"', '')]).toarray()
    return tfidf, vectorizer.get_feature_names_out()

def fasttext_embedding(text):
    ft = load_fasttext()
    words = text.replace('"', '').split()
    embeddings = [ft.get_word_vector(word) for word in words]
    return np.array(embeddings)

def glove_embedding(text):
    model = load_glove()
    words = text.replace('"', '').split()
    embeddings = [model[word] for word in words if word in model]
    return np.array(embeddings)

def bert_embedding(text):
    tokenizer, model = load_bert()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

def gpt2_embedding(text):
    tokenizer, model = load_gpt2()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# ===== Thực hiện nhúng nếu có dữ liệu =====

if processed_data:
    text = processed_data[0]  # Lấy dòng đầu tiên để thử nghiệm nhúng
    if embed_option == "Bag of N-grams":
        bag, feats = bag_of_ngrams(text)
        st.write("Bag of N-grams:")
        st.write(f"Features: {feats}")
        st.write(bag)
    elif embed_option == "TF-IDF":
        tfidf, feats = tfidf_vectorization(text)
        st.write("TF-IDF:")
        st.write(f"Features: {feats}")
        st.write(tfidf)
    elif embed_option == "FastText":
        emb = fasttext_embedding(text)
        st.write("FastText Embedding:")
        st.write(emb)
    elif embed_option == "GloVe":
        emb = glove_embedding(text)
        st.write("GloVe Embedding:")
        st.write(emb)
    elif embed_option == "BERT":
        emb = bert_embedding(text)
        st.write("BERT Embedding:")
        st.write(emb)
    elif embed_option == "GPT-2":
        emb = gpt2_embedding(text)
        st.write("GPT-2 Embedding:")
        st.write(emb)
else:
    st.warning("⚠️ Vui lòng nhập và xử lý văn bản trước khi nhúng.")

st.header("Bước 5: Huấn luyện và Dự đoán mô hình")

uploaded_file_5 = st.file_uploader("Tải file CSV dùng cho huấn luyện (riêng Bước 5)", type=["csv"], key="step5_upload")

if uploaded_file_5:
    try:
        try:
            df_5 = pd.read_csv(uploaded_file_5)
        except UnicodeDecodeError:
            df_5 = pd.read_csv(uploaded_file_5, encoding="ISO-8859-1")

        if df_5.empty:
            st.error("File CSV trống. Vui lòng kiểm tra lại.")
        else:
            st.success(f"Đã nạp {df_5.shape[0]} dòng, {df_5.shape[1]} cột.")
            st.dataframe(df_5.head())

            label_col = st.selectbox("Chọn cột nhãn (label)", df_5.columns, key="step5_label")
            text_col = st.selectbox("Chọn cột chứa văn bản", df_5.columns.drop(label_col), key="step5_text")

            if label_col and text_col:
                X = df_5[text_col].astype(str).tolist()
                y = df_5[label_col].astype(str).tolist()

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

                vec_type = st.selectbox("Vectorizer", ["CountVectorizer", "TF-IDF"], key="step5_vectorizer")
                vectorizer = CountVectorizer() if vec_type == "CountVectorizer" else TfidfVectorizer()
                X_train_vec = vectorizer.fit_transform(X_train)
                X_test_vec = vectorizer.transform(X_test)

                model_name = st.selectbox("Mô hình", ["Naive Bayes", "Logistic Regression", "Decision Tree", "KNN"], key="step5_model")

                if model_name == "Naive Bayes":
                    model = MultinomialNB()
                elif model_name == "Logistic Regression":
                    model = LogisticRegression(max_iter=500)
                elif model_name == "Decision Tree":
                    model = DecisionTreeClassifier(max_depth=5)
                elif model_name == "KNN":
                    model = KNeighborsClassifier(n_neighbors=5)

                if st.button("Huấn luyện mô hình", key="step5_train"):
                    model.fit(X_train_vec, y_train)
                    st.session_state['model'] = model
                    st.session_state['vectorizer'] = vectorizer

                    predictions = model.predict(X_test_vec)
                    acc = metrics.accuracy_score(y_test, predictions)

                    st.success(f"Đã huấn luyện mô hình {model_name}")
                    st.markdown(f"**🎯 Độ chính xác:** `{acc:.2%}`")

                    st.markdown("**📋 Báo cáo phân loại (rút gọn):**")
                    report = metrics.classification_report(y_test, predictions, output_dict=True)
                    report_df = pd.DataFrame(report).transpose().round(2).iloc[:-1]
                    st.dataframe(report_df)

                    # MA TRẬN NHẦM LẪN (BIỂU ĐỒ DUY NHẤT)
                    st.markdown("**Ma trận nhầm lẫn (biểu đồ):**")
                    cm = metrics.confusion_matrix(y_test, predictions)
                    labels = sorted(list(set(y_test)))
                    labels = [str(label) for label in labels]

                    fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                xticklabels=labels, yticklabels=labels, ax=ax_cm)
                    ax_cm.set_xlabel("Nhãn dự đoán")
                    ax_cm.set_ylabel("Nhãn thực tế")
                    ax_cm.set_title(f"Confusion Matrix - {model_name}")
                    st.pyplot(fig_cm)

                    # BIỂU ĐỒ PRECISION / RECALL / F1
                    st.markdown("**Biểu đồ Precision / Recall / F1-score:**")
                    fig_metrics, ax_metrics = plt.subplots(figsize=(8, 4))
                    report_df[['precision', 'recall', 'f1-score']].plot(kind='bar', ax=ax_metrics)
                    ax_metrics.set_ylim(0, 1)
                    ax_metrics.set_title(f"Chỉ số theo lớp - {model_name}")
                    ax_metrics.set_ylabel("Giá trị")
                    plt.xticks(rotation=45)
                    st.pyplot(fig_metrics)

            # DỰ ĐOÁN VĂN BẢN MỚI
            user_input = st.text_input("Nhập văn bản để mô hình dự đoán", key="step5_predict")
            if st.button("🔍 Dự đoán văn bản", key="step5_runpredict") and user_input:
                if 'model' in st.session_state and 'vectorizer' in st.session_state:
                    try:
                        transformed = st.session_state['vectorizer'].transform([user_input])
                        prediction = st.session_state['model'].predict(transformed)[0]
                        st.success(f"🔮 Dự đoán: **{prediction}**")
                    except Exception as e:
                        st.error(f"Lỗi khi dự đoán: {e}")
                else:
                    st.warning("⚠️ Cần huấn luyện mô hình trước khi dự đoán.")

    except Exception as e:
        st.error(f"Lỗi khi đọc file: {e}")
else:
    st.warning("⚠️ Vui lòng chọn file CSV hợp lệ để huấn luyện.")

st.header("Bước 6: Hệ thống Recommedation")

# Tải 2 file: 1 cho ratings, 1 cho movies
ratings_file = st.file_uploader("Tải file ratings.csv", type=["csv"], key="step6_ratings")
movies_file = st.file_uploader("Tải file movies.csv", type=["csv"], key="step6_movies")

@st.cache_data
def read_csv_file(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        return pd.read_csv(uploaded_file, encoding="ISO-8859-1")

@st.cache_data
def train_svd(ratings_pivot, k=50):
    U, S, Vt = np.linalg.svd(ratings_pivot, full_matrices=False)
    S_k = np.diag(S[:k])
    U_k = U[:, :k]
    Vt_k = Vt[:k, :]
    return U_k, S_k, Vt_k

def compute_v_new(Vt_k, movie_idx):
    return Vt_k[:, movie_idx]

def recommend_movies(movie_name, ratings_pivot, Vt_k, movies, top_n=5):
    movie_id = movies[movies['title'] == movie_name]['movieId']
    if movie_id.empty:
        return None, "Không tìm thấy phim này!"

    movie_id_value = movie_id.values[0]
    if movie_id_value not in ratings_pivot.index:
        return None, "Phim đã được nạp nhưng chưa có rating."

    movie_idx = ratings_pivot.index.get_loc(movie_id_value)
    V_new = compute_v_new(Vt_k, movie_idx)
    similarity_scores = cosine_similarity(Vt_k.T, V_new.reshape(1, -1))[:, 0]

    similar_indices = np.argsort(-similarity_scores)[1:top_n+1]
    valid_indices = [i for i in similar_indices if i < len(ratings_pivot)]
    recommended_ids = ratings_pivot.index[valid_indices].tolist()
    recommended_movies = movies[movies['movieId'].isin(recommended_ids)]
    return recommended_movies, None

if ratings_file and movies_file:
    ratings = read_csv_file(ratings_file)
    movies = read_csv_file(movies_file)

    if ratings.empty or movies.empty:
        st.error("⚠️ File rỗng hoặc không đọc được.")
    else:
        max_samples = st.slider("Số lượng ratings tối đa", 1000, 100000, 5000, step=1000)
        if len(ratings) > max_samples:
            ratings = ratings.sample(n=max_samples, random_state=42)

        ratings_pivot = ratings.pivot(index='movieId', columns='userId', values='rating').fillna(0)
        U_k, S_k, Vt_k = train_svd(ratings_pivot)

        selected_movie = st.text_input("🔍 Nhập tên phim:", "")
        if st.button("🎮 Gợi ý phim") and selected_movie:
            if selected_movie in movies['title'].values:
                movie_details = movies[movies['title'] == selected_movie].iloc[0]
                st.subheader(f"Thông tin phim: {selected_movie}")
                st.write(f"🎭 **Thể loại:** {movie_details['genres']}")
                recommendations, error_message = recommend_movies(selected_movie, ratings_pivot, Vt_k, movies)

                if isinstance(recommendations, pd.DataFrame) and not recommendations.empty:
                    st.subheader("🎬 Phim gợi ý:")
                    for _, row in recommendations.iterrows():
                        st.markdown(f"🎥 **{row['title']}**  🎭 {row['genres']}")
                elif error_message:
                    st.warning(error_message)
            else:
                st.error("Tên phim không tồn tại trong danh sách.")
else:
    st.info("Vui lòng tải đủ 2 file: `ratings.csv` và `movies.csv`.")
    

st.header("Bước 7: Chatbot")

import unicodedata
from sklearn.pipeline import make_pipeline

# Hàm xử lý tiếng Việt không dấu
def remove_accents(text):
    return ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn').lower()

try:
    df_players = pd.read_csv("attacking.csv")
    df_intents = pd.read_csv("intents.csv")

    model_chat = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model_chat.fit(df_intents['question'], df_intents['intent'])

    # Giao diện hội thoại
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    def chatbot_response(user_input):
        predicted_intent = model_chat.predict([user_input])[0]
        normalized_input = remove_accents(user_input)

        for _, row in df_players.iterrows():
            player_norm = remove_accents(row['player_name'])
            if player_norm in normalized_input:
                value = row[predicted_intent]
                if predicted_intent == "assists":
                    return f"{row['player_name']} đã kiến tạo {value} lần."
                elif predicted_intent == "match_played":
                    return f"{row['player_name']} đã thi đấu {value} trận."
                elif predicted_intent == "dribbles":
                    return f"{row['player_name']} đã rê bóng {value} lần."
                elif predicted_intent == "corner_taken":
                    return f"{row['player_name']} đã đá {value} quả phạt góc."
                elif predicted_intent == "offsides":
                    return f"{row['player_name']} đã việt vị {value} lần."
                elif predicted_intent == "position":
                    return f"{row['player_name']} chơi ở vị trí {value}."
                elif predicted_intent == "club":
                    return f"{row['player_name']} thuộc câu lạc bộ {value}."
        return "❓ Tôi không hiểu. Hãy hỏi như: 'Ronaldo đá vị trí nào?' hoặc 'Messi rê bóng bao nhiêu lần?'"

    with st.form("chat_form"):
        user_input = st.text_input("Bạn", placeholder="Nhập câu hỏi về cầu thủ...")
        submitted = st.form_submit_button("Gửi")

    if submitted and user_input:
        response = chatbot_response(user_input)
        st.session_state.chat_history.append((user_input, response))

    for user, bot in reversed(st.session_state.chat_history):
        st.markdown(f"<div style='background-color:#d1ecf1; padding:8px; border-radius:5px; margin:4px'><b>Bạn:</b> {user}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='background-color:#fff3cd; padding:8px; border-radius:5px; margin:4px'><b>Bot:</b> {bot}</div>", unsafe_allow_html=True)

except Exception as e:
    st.error(f"⚠️ Không thể load chatbot: {e}")