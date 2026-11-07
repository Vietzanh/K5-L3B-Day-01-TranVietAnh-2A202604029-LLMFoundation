# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202604029
- **Họ và tên:** Trần Việt Anh
- **SEED** (in ở ô đầu notebook): 5639

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☐ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ |

## 2. Kết quả chính

                                 precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            1.000   0.753      0.859        0.824   
2. BoW + MLP (deep learning)         1.000   0.887      0.940        0.889   
3a. Encoder (MiniLM) + kNN           0.913   0.969      0.940        0.976   
3b. Encoder (MiniLM) + LogReg        0.968   0.948      0.958        0.974   
3c. Encoder (MiniLM) + MLP head      0.958   0.938      0.948        0.974   
4a. LLM nhỏ zero-shot                0.952   1.000      0.976        0.976   
4c. LLM nhỏ few-shot                 0.952   1.000      0.976        0.976   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)           10.456      13876.853       0.063   
2. BoW + MLP (deep learning)        13.463       9279.806       0.360   
3a. Encoder (MiniLM) + kNN         100.403         18.883      55.505   
3b. Encoder (MiniLM) + LogReg      100.899         26.543      27.293   
3c. Encoder (MiniLM) + MLP head    101.419         26.439      42.867   
4a. LLM nhỏ zero-shot                0.000          1.586     542.466   
4c. LLM nhỏ few-shot                 0.000          0.490    1740.144   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.287         0.000  
2. BoW + MLP (deep learning)          1.710         0.000  
3a. Encoder (MiniLM) + kNN           66.992         0.000  
3b. Encoder (MiniLM) + LogReg        37.999         0.000  
3c. Encoder (MiniLM) + MLP head      62.441         0.000  
4a. LLM nhỏ zero-shot              1114.824         0.032  
4c. LLM nhỏ few-shot               3144.810         0.065  


## 3. Dự đoán và thực tế

- Dự đoán [mr_lr]: dự đoán trường hợp lr=50 sẽ cho loss có hình zic zac mà không hội tụ, kết quả cho thấy model hội tụ sau 2 epoc. Vì với lr = 50, sau 2 epoch, loss may mắn đạt gần đến cực tiểu; điều này không khẳng định lr = 50 là tốt mà vì loss sau 2 epoch đã nhỏ kéo theo gradient nhỏ nên dù lr lớn cũng chưa chắc kéo được trọng số thay đổi quá nhiều.
- Dự đoán [emb_search]: dự đoán nếu không chứa các từ spam quen thuộc như "free", "win", "prize" thì sẽ trả về ham, trong khi đó kết quả hầu hết là spam. Vì encoder đã được huấn luyện để hiểu được ở một mức độ nào đó mối quan hệ giữa các từ trong câu chứ không chỉ dựa trên sự xuất hiện của các từ khóa, nên vẫn có khả năng đưa ra kết quả tốt dù trong câu không chứa các từ khóa spam quen thuộc.
- Dự đoán [emb_tricky]: trong trường hợp các tin né từ khóa spam hoặc chứa từ khóa nhưng vô hại ham, dự đoán encoder + LogReg dự đoán đúng nhiều hơn, kết quả ra cả 6 mẫu thì BoW và embedding đều giống nhau, tức là hòa. Có thể embedding thực sự tốt hơn trong trường hợp này nhưng chỉ với 6 mẫu thì chưa đủ để kết luận model nào tốt hơn.

## 4. Phân tích 50 case khó

Model tụt nhiều nhất là 1. BoW + LogReg, F1 giảm từ 0.824 xuống 0.286. 2 nhóm case khiến BoW + LogReg thất bại là Spam giọng tự nhiên và Né từ khóa.
- Trong Spam giọng tự nhiên, tin nhắn dùng ngôn ngữ bình thường mà không có các từ khóa spam. BoW + LogReg đánh giá dựa trên sự xuất hiện của từ khóa nên dễ đánh giá những câu chứa từ tự nhiên là ham.
- Trong Né từ khóa, tin nhắn cũng không chứa các từ khóa spam, nên cơ chế khiến model sai giống với Spam giọng tự nhiên.

## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*

## 6. Phương án cho sản phẩm thật

- Đánh giá: Dựa vào bảng so sánh, có 3 model đạt yêu cầu p95 < 50ms: BoW + LogReg, BoW + MLP, Emb + LogReg; trong đó chỉ có 2 model đạt 116 tin/s: BoW + LogReg, BoW + MLP. Trong 2 model này, BoW + MLP đạt F1 cao hơn. Ước tính chi phí dùng LLM zero-shot, chi phí trên 1000 SMS là 0.032 USD, Cost = 10_000_000 * 30 / 1000 * 0.032 = 9600 USD/tháng.
- Phương án: thiết kế gồm tầng 1 là BoW + LogReg xử lý nhanh, throughput cao, nếu kết quả xác suất cao p < 0.3 hoặc p > 0.7 thì chọn kết quả model này. Nếu kết quả lưỡng lự p thuộc 0.3 - 0.7 thì chuyển sang tầng 2 LLM để suy luận vào đạt kết quả tốt hơn. Cách này tốt hơn vì nó vừa đạt độ chính xác tương đối ổn khi dùng model nhẹ với các case dễ, tiết kiệm chi phí tính toán và thời gian khi chỉ dùng model lớn với những case khó.

## 7. AI log và điều đã học

Các câu hỏi AI:
- Tại sao các mô hình có kết quả tốt trên tập test thông thường nhưng lại thất bại trên 50 case khó?
- Cách sửa build_classification_prompt để LLM trả về JSON gồm label và reason, đồng thời viết hàm parse JSON.
- Cách chạy lại hai phương pháp trên cùng 10 tin SUB và so sánh token, chi phí.
- Với yêu cầu nhà mạng xử lý 10 triệu SMS/ngày và p95 dưới 50 ms, nên chọn mô hình nào? Có thể thiết kế hệ thống kết hợp không?

Bài học rút ra: accuracy hoặc F1 cao chưa chắc đã là lựa chọn tốt nhất, mà thực tế còn phải xem xét latency, throughput và chi phí. Việc sử dụng LLM không chỉ phụ thuộc prompt mà còn cần kiểm soát parser định dạng đầu ra. Hệ thống không nhất thiết phải dùng model mạnh nhất cho mọi dữ liệu mà có thể kết hợp nhiều model để tận dụng tốc độ của model đơn giản và chất lượng tốt hơn của model mạnh.
