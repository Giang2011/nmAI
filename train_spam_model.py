import os
import pandas as pd
import numpy as np


# =========================================================
# PHẦN 1. ĐỌC DỮ LIỆU VÀ CHUẨN BỊ MA TRẬN TRAIN
# - emails.csv đã có sẵn 3000 đặc trưng (3000 từ phổ biến nhất)
# - Cột đầu là Email No. chỉ để định danh nên sẽ bỏ đi
# - Cột cuối Prediction là nhãn: 1 = spam, 0 = không spam
# =========================================================
DATA_PATH = "emails.csv"
MODEL_PATH = "spam_model.npz"


def load_dataset(path):
    df = pd.read_csv(path)
    df = df.drop(columns=["Email No."])

    X = df.drop(columns=["Prediction"]).values
    y = df["Prediction"].values
    feature_names = df.drop(columns=["Prediction"]).columns.to_list()
    return X, y, feature_names


# =========================================================
# PHẦN 2. XÂY DỰNG MÔ HÌNH MULTINOMIAL NAIVE BAYES
# - Ta chỉ cần lưu các tham số sau khi train xong
# - Lần sau chạy lại có thể load từ file thay vì train lại
# =========================================================
class MultinomialNaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Đếm số email spam / ham
        n_spam = np.sum(y == 1)
        n_ham = np.sum(y == 0)

        # Xác suất tiên nghiệm ở dạng log
        self.log_prior_spam = np.log(n_spam / n_samples)
        self.log_prior_ham = np.log(n_ham / n_samples)

        # Đếm tổng số lần xuất hiện của mỗi từ trong spam và ham
        spam_words_count = np.sum(X[y == 1], axis=0)
        ham_words_count = np.sum(X[y == 0], axis=0)

        total_spam_words = np.sum(spam_words_count)
        total_ham_words = np.sum(ham_words_count)

        # Laplace smoothing để tránh xác suất bằng 0
        self.log_prob_spam = np.log((spam_words_count + 1) / (total_spam_words + n_features))
        self.log_prob_ham = np.log((ham_words_count + 1) / (total_ham_words + n_features))

    def predict(self, X):
        predictions = []
        for x in X:
            score_spam = self.log_prior_spam + np.sum(x * self.log_prob_spam)
            score_ham = self.log_prior_ham + np.sum(x * self.log_prob_ham)
            predictions.append(1 if score_spam > score_ham else 0)
        return np.array(predictions)

    # =========================================================
    # PHẦN 3. LƯU / TẢI THAM SỐ MODEL
    # - Chỉ lưu tham số, không cần lưu toàn bộ object
    # - Dùng np.savez để không phụ thuộc thư viện ngoài
    # =========================================================
    def save(self, path, feature_names):
        np.savez(
            path,
            log_prior_spam=self.log_prior_spam,
            log_prior_ham=self.log_prior_ham,
            log_prob_spam=self.log_prob_spam,
            log_prob_ham=self.log_prob_ham,
            feature_names=np.array(feature_names, dtype=object),
        )

    def load(self, path):
        data = np.load(path, allow_pickle=True)
        self.log_prior_spam = data["log_prior_spam"]
        self.log_prior_ham = data["log_prior_ham"]
        self.log_prob_spam = data["log_prob_spam"]
        self.log_prob_ham = data["log_prob_ham"]
        self.feature_names = data["feature_names"].tolist()


# =========================================================
# PHẦN 4. TRAIN NẾU CHƯA CÓ MODEL, NGƯỢC LẠI LOAD LẠI
# - Đây là file train chính
# - Chạy file này để tạo model.npz
# =========================================================
def main():
    print("Đang đọc dữ liệu...")
    X, y, feature_names = load_dataset(DATA_PATH)

    # Tách train/test thủ công để tự đánh giá mô hình
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    train_size = int(0.8 * len(X))
    train_idx, test_idx = indices[:train_size], indices[train_size:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    model = MultinomialNaiveBayes()

    if os.path.exists(MODEL_PATH):
        print(f"Đã tìm thấy file model: {MODEL_PATH}. Đang tải lại tham số...")
        model.load(MODEL_PATH)
    else:
        print("Chưa có model lưu sẵn. Đang huấn luyện mô hình...")
        model.fit(X_train, y_train)
        model.save(MODEL_PATH, feature_names)
        print(f"Đã lưu model vào {MODEL_PATH}")

    print("Đang dự đoán trên tập test...")
    predictions = model.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print(f"Hoàn tất! Accuracy của mô hình: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
