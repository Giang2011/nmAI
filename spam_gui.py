import os
import re
import tkinter as tk
from tkinter import messagebox

import numpy as np
import pandas as pd


# =========================================================
# PHẦN 1. CÁC THIẾT LẬP CHUNG
# - File model sau khi train sẽ được lưu ở đây
# - Ta sẽ load model + vocabulary từ file npz
# =========================================================
DATA_PATH = "emails.csv"
MODEL_PATH = "spam_model.npz"


# =========================================================
# PHẦN 2. HÀM TIỀN XỬ LÝ VĂN BẢN
# - lowercase
# - bỏ dấu câu / ký tự không phải chữ cái
# - tách khoảng trắng thành token
# - nếu từ không có trong vocab thì bỏ qua
# =========================================================
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split() if text else []


# =========================================================
# PHẦN 3. TẠO VECTOR ĐẦU VÀO TỪ TEXT NGƯỜI DÙNG
# - Convert danh sách token sang vector đếm từ
# - Chỉ đếm các từ có trong vocabulary train
# =========================================================
def text_to_vector(text, vocab_index, n_features):
    tokens = preprocess_text(text)
    vector = np.zeros(n_features, dtype=np.float64)

    for token in tokens:
        idx = vocab_index.get(token)
        if idx is not None:
            vector[idx] += 1

    return vector


# =========================================================
# PHẦN 4. LỚP NAIVE BAYES CHO DỰ ĐOÁN
# - Chỉ cần load tham số đã train từ file
# - Không train lại trong giao diện
# =========================================================
class MultinomialNaiveBayes:
    def load(self, path):
        data = np.load(path, allow_pickle=True)
        self.log_prior_spam = data["log_prior_spam"]
        self.log_prior_ham = data["log_prior_ham"]
        self.log_prob_spam = data["log_prob_spam"]
        self.log_prob_ham = data["log_prob_ham"]
        self.feature_names = data["feature_names"].tolist()

    def predict_scores(self, x):
        score_spam = self.log_prior_spam + np.sum(x * self.log_prob_spam)
        score_ham = self.log_prior_ham + np.sum(x * self.log_prob_ham)
        return score_spam, score_ham

    def predict_proba_one(self, x):
        score_spam, score_ham = self.predict_scores(x)
        max_score = max(score_spam, score_ham)
        exp_spam = np.exp(score_spam - max_score)
        exp_ham = np.exp(score_ham - max_score)
        total = exp_spam + exp_ham
        prob_spam = exp_spam / total
        prob_ham = exp_ham / total
        return prob_spam, prob_ham

    def predict_one(self, x):
        score_spam, score_ham = self.predict_scores(x)
        return 1 if score_spam > score_ham else 0


# =========================================================
# PHẦN 5. LOAD MODEL VÀ XÂY DỰNG VOCAB INDEX
# - feature_names chính là 3000 từ của dataset
# - Dùng dict để tra từ -> vị trí trong vector
# =========================================================
def load_model_and_vocab():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Không tìm thấy {MODEL_PATH}. Hãy chạy file train_spam_model.py trước."
        )

    model = MultinomialNaiveBayes()
    model.load(MODEL_PATH)
    vocab_index = {word: i for i, word in enumerate(model.feature_names)}
    return model, vocab_index, len(model.feature_names)


# =========================================================
# PHẦN 6. TẠO GIAO DIỆN GUI
# - Người dùng nhập nội dung email
# - Bấm nút để dự đoán spam / ham
# - Kết quả hiển thị ngay trên giao diện
# =========================================================
def main():
    try:
        model, vocab_index, n_features = load_model_and_vocab()
    except FileNotFoundError as e:
        messagebox.showerror("Thiếu model", str(e))
        return

    root = tk.Tk()
    root.title("Spam Detector")
    root.geometry("760x520")
    root.resizable(False, False)

    # Khung tiêu đề
    title = tk.Label(
        root,
        text="Spam Email Detector",
        font=("Arial", 20, "bold"),
        pady=12,
    )
    title.pack()

    # Hướng dẫn nhập liệu
    instruction = tk.Label(
        root,
        text=(
            "Nhập nội dung email bên dưới. Hệ thống sẽ tự lowercase, bỏ dấu câu, "
            "tách từ theo khoảng trắng, rồi đối chiếu với vocabulary đã train."
        ),
        font=("Arial", 10),
        wraplength=700,
        justify="left",
        pady=8,
    )
    instruction.pack()

    # Ô nhập email
    text_frame = tk.Frame(root)
    text_frame.pack(padx=16, pady=10, fill="both", expand=False)

    text_box = tk.Text(text_frame, height=14, width=88, font=("Arial", 11), wrap="word")
    text_box.pack(side="left", fill="both", expand=True)

    scrollbar = tk.Scrollbar(text_frame, command=text_box.yview)
    scrollbar.pack(side="right", fill="y")
    text_box.config(yscrollcommand=scrollbar.set)

    # Vùng hiển thị kết quả
    result_var = tk.StringVar(value="Kết quả sẽ hiển thị ở đây...")
    result_label = tk.Label(
        root,
        textvariable=result_var,
        font=("Arial", 14, "bold"),
        pady=18,
        fg="#1f2937",
    )
    result_label.pack()

    # Hàm xử lý dự đoán
    def predict_email():
        email_text = text_box.get("1.0", tk.END).strip()

        if not email_text:
            messagebox.showwarning("Thiếu dữ liệu", "Vui lòng nhập nội dung email.")
            return

        # Chuyển text sang vector đếm từ theo vocabulary train
        vector = text_to_vector(email_text, vocab_index, n_features)

        # Dù không có từ nào khớp vocabulary thì vẫn chạy Naive Bayes.
        # Khi vector toàn 0, kết quả sẽ phụ thuộc vào prior spam/ham.
        prob_spam, prob_ham = model.predict_proba_one(vector)
        pred = 1 if prob_spam > prob_ham else 0

        if pred == 1:
            result_var.set(
                f"Kết quả: SPAM | % spam: {prob_spam * 100:.2f}% | % ham: {prob_ham * 100:.2f}%"
            )
            result_label.config(fg="#b91c1c")
        else:
            result_var.set(
                f"Kết quả: HAM | % ham: {prob_ham * 100:.2f}% | % spam: {prob_spam * 100:.2f}%"
            )
            result_label.config(fg="#15803d")

    # Nút predict
    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=8)

    predict_btn = tk.Button(
        btn_frame,
        text="Dự đoán",
        font=("Arial", 12, "bold"),
        bg="#2563eb",
        fg="white",
        padx=18,
        pady=8,
        command=predict_email,
    )
    predict_btn.pack(side="left", padx=8)

    clear_btn = tk.Button(
        btn_frame,
        text="Xóa nội dung",
        font=("Arial", 12),
        padx=18,
        pady=8,
        command=lambda: (text_box.delete("1.0", tk.END), result_var.set("Kết quả sẽ hiển thị ở đây..."), result_label.config(fg="#1f2937")),
    )
    clear_btn.pack(side="left", padx=8)

    # Thông tin thêm
    footer = tk.Label(
        root,
        text="Mô hình dùng 3000 từ phổ biến nhất trong emails.csv và lưu tham số sau khi train. Kết quả hiển thị xác suất tương đối giữa spam và ham.",
        font=("Arial", 9),
        fg="#6b7280",
        pady=8,
    )
    footer.pack(side="bottom")

    root.mainloop()


if __name__ == "__main__":
    main()
