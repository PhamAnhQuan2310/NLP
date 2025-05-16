# Ứng dụng Xử Lý Ngôn Ngữ Tự Nhiên (NLP)

Đây là một ứng dụng Web xây dựng bằng **Streamlit**, hỗ trợ xử lý xủ lý ngôn ngữ tự nhiên với các chức năng:
- Nhập dữ liệu từ tay, file CSV hoặc từ trang web
- Tăng cường và tiền xử lý dữ liệu văn bản
- Vector hóa và nhúng văn bản bằng nhiều kỹ thuật 
- Huấn luyện và đánh giá các mô hình học máy
- Hệ thống gợi ý phim dựa trên sở thích
- Chatbot hỏi đáp thông tin cầu thủ bóng đá


## Công nghệ sử dụng

- Python, Streamlit
- Scikit-learn, NLTK, SpaCy, Gensim, FastText
- Mô hình nhúng từ: GloVe, BERT, GPT-2 (Transformers)
- BeautifulSoup (cào dữ liệu web)
- Matplotlib, Seaborn


## Các chức năng chính

### 1. Nhập dữ liệu
- Nhập tay
- Tải file CSV
- Cào nội dung từ URL

### 2. Tăng cường dữ liệu
- Đảo ngược câu
- Thay từ đồng nghĩa
- Thay thực thể
- Thêm ký tự nhiễu

### 3. Tiền xử lý văn bản
- Loại bỏ dấu câu, stopwords
- Tách từ, stemming, lemmatization
- POS tagging (gắn nhãn từ loại)

### 4. Vector hóa
- One-Hot Encoding
- Bag of Words (BoW)

### 5. Nhúng văn bản
- Bag of N-grams
- TF-IDF
- FastText
- GloVe
- BERT
- GPT-2

### 6. Huấn luyện mô hình
- Chọn tập dữ liệu và cột label
- Vector hóa bằng CountVectorizer hoặc TF-IDF
- Huấn luyện mô hình: Naive Bayes, Logistic Regression, Decision Tree, KNN
- Đánh giá: độ chính xác, báo cáo phân loại, biểu đồ, ma trận nhầm lẫn
- Dự đoán đoạn văn nhập vào

### 7. Hệ thống gợi ý phim (Recommendation)
- Dựa trên dữ liệu `ratings.csv` và `movies.csv`
- Sử dụng phương pháp SVD để giảm chiều
- Gợi ý top phim tương tự một phim đã chọn (product base)

### 8. Chatbot cầu thủ bóng đá

## chạy ứng dụng: streamlit run BTCaNhan.py


