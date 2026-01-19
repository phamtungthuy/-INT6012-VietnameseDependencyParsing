# Phân tích Dữ liệu Dependency Parsing

## 1. Tổng quan Dữ liệu (Overview)

Thống kê cơ bản về 3 tập dữ liệu Train, Dev, và Test.

| Metric                     | Train     | Dev       | Test      |
| :------------------------- | :-------- | :-------- | :-------- |
| **Số câu (Sentences)**     | 1400      | 1123      | 800       |
| **Số từ (Tokens)**         | 20,215    | 26,162    | 11,692    |
| **Chiều dài TB (Avg Len)** | 14.44     | **23.30** | 14.62     |
| **Max Length**             | 25        | 96        | 25        |
| **Non-Projective**         | 0.36% (5) | 0.80% (9) | 0.13% (1) |
| **Avg Dependency Dist**    | 2.77      | 3.24      | 2.86      |

### Nhận xét & Ý nghĩa:

- **Kích thước tập Train nhỏ**: Chỉ có 1400 câu (~20k từ). Đây là tập dữ liệu rất nhỏ cho các mô hình Deep Learning phức tạp.
  - **Implication**: Cần sử dụng **Pretrained Embeddings** (như PhoBERT hoặc FastText) để tránh overfitting. Regularization (Dropout) cần được thiết lập cao (ví dụ: `0.33` - `0.5`).
- **Sự chênh lệch giữa Dev và Train**: Tập Dev có số lượng từ **lớn hơn** tập Train (26k vs 20k) và chiều dài câu trung bình **cao hơn đáng kể** (23.3 vs 14.4).
  - **Implication**: Mô hình học trên câu ngắn (Train) có thể gặp khó khăn khi gặp câu dài và phức tạp trong tập Dev. Cần chú ý đánh giá kỹ trên Dev để đảm bảo khả năng tổng quát hóa độ dài (length generalization). Max length trong Dev lên tới 96, trong khi Train chỉ 25. Cần xem lại cách chia dữ liệu hoặc chấp nhận bias này.
- **Cấu trúc Non-projective thấp**: Tỉ lệ cây không hợp lệ (non-projective) rất thấp (< 1%).
  - **Implication**: Giải thuật Eisner (Projective) sẽ hoạt động tốt và nhanh. Chu-Liu-Edmonds (Non-Projective) cũng tốt nhưng không mang lại lợi thế vượt trội về độ chính xác cấu trúc so với Eisner, tuy nhiên vẫn cần thiết để bao phủ 100% các trường hợp.

## 2. Phân bố Nhãn (Label Distribution)

### 2.1. POS Tags (Top 10)

| POS       | Train        | Dev          | Test         |
| :-------- | :----------- | :----------- | :----------- |
| **NOUN**  | 5617 (27.8%) | 7843 (30.0%) | 3029 (25.9%) |
| **VERB**  | 3821 (18.9%) | 4913 (18.8%) | 2134 (18.3%) |
| **PUNCT** | 2931 (14.5%) | 3517 (13.4%) | 1703 (14.6%) |
| **ADV**   | 1481 (7.3%)  | 1643 (6.3%)  | 1020 (8.7%)  |
| **ADP**   | 1207 (6.0%)  | 1540 (5.9%)  | 645 (5.5%)   |
| **ADJ**   | 1097 (5.4%)  | 1564 (6.0%)  | 719 (6.1%)   |
| **PROPN** | 895 (4.4%)   | 1060 (4.1%)  | 560 (4.8%)   |
| **PRON**  | 803 (4.0%)   | 923 (3.5%)   | 531 (4.5%)   |
| **NUM**   | 622 (3.1%)   | 937 (3.6%)   | 269 (2.3%)   |
| **SCONJ** | 501 (2.5%)   | 628 (2.4%)   | 382 (3.3%)   |

### 2.2. Dependency Relations (Top 10)

| DepRel       | Train | Dev  | Test |
| :----------- | :---- | :--- | :--- |
| **punct**    | 2931  | 3517 | 1703 |
| **obj**      | 1660  | 1835 | 867  |
| **nsubj**    | 1577  | 1519 | 958  |
| **case**     | 1121  | 1460 | 565  |
| **root**     | 1400  | 1123 | 800  |
| **advmod**   | 1094  | 1342 | 752  |
| **compound** | 520   | 1288 | 154  |
| **conj**     | 745   | 1259 | 468  |
| **nmod**     | 740   | 1304 | 461  |
| **xcomp**    | 710   | 917  | 329  |

### Nhận xét & Ý nghĩa:

- **Nhãn phức tạp thấp**: Các nhãn như `acl:subj`, `obl:tmod`, `clf:det` xuất hiện với tần suất trung bình. Mô hình cần phân biệt tốt các "subtype" (dấu hai chấm) này.
- **Implication**:
  - Hệ thống Relation Scorer cần đủ mạnh để phân biệt các nhãn mịn (fine-grained). Kiến trúc **Triaffine** (như đã triển khai) với các MLP riêng biệt cho Relation Head/Dep là phù hợp để nắm bắt các tương tác tinh tế này.
  - Sự mất cân bằng nhãn (`punct`, `obj`, `nsubj` chiếm đa số) có thể gợi ý sử dụng **Focal Loss** hoặc **Label Smoothing** nếu mô hình gặp khó khăn với các nhãn hiếm (như `vocative`, `flat`).

## 3. Khoảng cách phụ thuộc (Dependency Distance)

- **Trung bình**: ~2.8 - 3.2 từ.
- **Dev Set**: Khoảng cách trung bình dài hơn (3.24) do câu dài hơn.

### Ý nghĩa:

- Khoảng cách phụ thuộc ngắn cho thấy ngữ pháp tiếng Việt (trong tập này) có xu hướng cục bộ (local attachments).
- Tuy nhiên, các quan hệ xa (như `root` đến cuối câu, hoặc `conj` trong câu ghép) vẫn tồn tại. BiLSTM với khả năng ghi nhớ dài hạn (Long Short-Term Memory) là lựa chọn mô hình phù hợp.

## 4. Kiến nghị cho Mô hình (Recommendations)

1.  **Chiến lược Encoder**:
    - Do dữ liệu Train nhỏ và Dev lệch phân phối (OOD về độ dài), việc sử dụng **BERT/PhoBERT** (multilingual hoặc vietnamese) là cực kỳ quan trọng để cung cấp ngữ nghĩa phong phú mà BiLSTM thuần có thể không học hết được từ lượng dữ liệu ít ỏi.
    - Nếu dùng BiLSTM thuần: Cần dropout cao và có thể cân nhắc giảm số chiều hidden nếu thấy overfitting.

2.  **Cấu trúc Decoder**:
    - **Arc Scorer**: Biaffine là chuẩn mực.
    - **Relation Scorer**: Dữ liệu có nhiều nhãn mịn (subtypes), việc nâng cấp lên **Triaffine** (sử dụng thêm Relationship Embeddings) như yêu cầu là hợp lý để tăng độ chính xác phân loại nhãn. Tương tác 3 chiều (Head, Dep, Label) giúp mô hình "hiểu" nhãn nào phù hợp với cặp từ vựng cụ thể hơn.

3.  **Regularization**:
    - Cần áp dụng **Word Dropout** (thay thế từ bằng UNK ngẫu nhiên) để mô hình không phụ thuộc quá nhiều vào từ vựng cụ thể trong tập Train nhỏ.

4.  **Training**:
    - Cần Early Stopping dựa trên LAS của tập Dev. Do Dev khó hơn Train, LAS trên Dev có thể thấp hơn đáng kể.

5.  **Xử lý Non-projective**:
    - Mặc dù tỉ lệ thấp, nên sử dụng thuật toán **MST (Chu-Liu-Edmonds)** khi decode để đảm bảo tính tổng quát. Nếu cần tốc độ cực cao thì mới dùng Eisner.
