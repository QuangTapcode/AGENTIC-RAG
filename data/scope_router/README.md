# Scope Router

Scope Router chạy trước retrieval:

1. Chuẩn hóa query và phát hiện ngôn ngữ `vi`/`en`.
2. Chấm điểm tín hiệu y tế và tín hiệu ngoài phạm vi bằng keyword/manifest terms.
3. Nếu `in_scope`, pipeline được phép sang retrieval.
4. Nếu `out_of_scope`, router gọi tool terminal `reject_out_of_scope` và dừng pipeline.
5. Nếu tín hiệu không đủ hoặc mâu thuẫn, router yêu cầu người dùng làm rõ.

`ScopeRouter.route()` nhận thêm `llm_classifier` tùy chọn. Adapter LLM chỉ được dùng khi rule-based confidence thấp/mơ hồ; kết quả phải trả về `decision` hợp lệ (`in_scope`, `out_of_scope`, `clarify`) và confidence tối thiểu 0.60.

## Chạy

```powershell
..\.venv\Scripts\python.exe -m unittest discover -s tests -v
..\.venv\Scripts\python.exe src\scope_router\scope_router.py "Bệnh tiểu đường có triệu chứng gì?"
..\.venv\Scripts\python.exe src\scope_router\scope_router.py "Viết code Python" --log-file data\scope_router\router.jsonl
```

Log feedback có event `router_misclassification` với expected, actual, root cause, fix và lesson learned. Không bật log query y tế trong production nếu chưa có chính sách bảo vệ dữ liệu.
