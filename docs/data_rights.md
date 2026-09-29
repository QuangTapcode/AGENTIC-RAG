# Data rights, attribution và điều khoản tái phân phối

## Nguồn dataset

Prototype dùng **30 WHO fact sheets** (bản Markdown được trích xuất từ trang
`who.int/news-room/fact-sheets`). Manifest đầy đủ trong
[data/manifest.json](../data/manifest.json).

- Chủ thể phát hành: World Health Organization (WHO).
- Catalogue gốc: https://www.who.int/news-room/fact-sheets
- Ngày thu thập: `2026-09-27` (ghi tại `manifest.dataset_id`).
- Ngôn ngữ nguồn: English (`source_language=en`).

## Attribution bắt buộc

Mỗi câu trả lời và mỗi report có citation phải chứa:

1. Tên tài liệu (`title`) — ví dụ *"Cancer"*.
2. `source_url` gốc trên who.int.
3. Section header khi có.
4. Khi thiếu page, dùng `page unavailable` + `source_line_start/end`.

Trong prompt và citation renderer, các yêu cầu này đã được hard-coded ở
[answer_generator.py:80-92](../src/answering/answer_generator.py#L80-L92).

## Điều khoản WHO cần lưu ý

WHO công bố nội dung fact sheet với điều khoản riêng (CC BY-NC-SA 3.0 IGO cho
phần lớn nội dung web, kèm ngoại lệ với logo/emblem và một số dữ liệu bên thứ ba).
Tóm tắt các nghĩa vụ chính:

- **Attribution**: giữ nguồn gốc WHO và link tới URL gốc.
- **NonCommercial**: không dùng trực tiếp cho mục đích thương mại nếu không có
  giấy phép riêng từ WHO.
- **ShareAlike**: nếu redistribute bản phái sinh, phải dùng cùng license hoặc
  tương thích.
- Không được dùng logo/emblem của WHO nếu chưa có phép.
- Không tạo ấn tượng WHO endorse hệ thống của bạn.

Trang tham khảo (đọc trước khi public dataset hoặc dùng thương mại):

- Điều khoản chung của WHO: https://www.who.int/about/policies/publishing/copyright
- Điều khoản re-use dữ liệu WHO: https://www.who.int/about/policies/publishing/permissions

## Trạng thái cho prototype này

- **Nội bộ / học thuật / demo cá nhân**: ✅ cho phép với điều kiện giữ
  attribution và link nguồn.
- **Public dataset (upload thẳng lên HuggingFace / GitHub release binary)**: ⛔
  **cần confirm quyền** từ WHO trước khi publish `data/raw/*.md`. Prototype
  hiện chỉ commit `manifest.json` + `parsed/*.json` với reference đến
  `source_url`; các file raw có thể được rebuild bằng script fetch (không lưu
  binary trong repo public).
- **Thương mại**: ⛔ chưa được cấp phép; **không** dùng trực tiếp cho SaaS trước
  khi có license từ WHO.

## Checklist trước khi share/publish

- [ ] Có giữ `source_url` trong mọi citation và ở `manifest.json`?
- [ ] Có ghi WHO là nguồn gốc trong README/demo?
- [ ] Đã kiểm tra không dùng logo/emblem WHO trong UI?
- [ ] Nếu redistribute `data/raw/*.md`: đã đọc và tuân thủ điều khoản WHO
  copyright/permissions và có ghi rõ license phái sinh (CC BY-NC-SA 3.0 IGO)?
- [ ] Nếu dùng thương mại: đã có confirmation/permission từ WHO?

Nếu **bất kỳ** ô trên chưa tick, mặc định coi dataset là **internal only**.
