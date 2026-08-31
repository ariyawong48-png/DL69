# การบ้านสัปดาห์ที่ 9 — PyTorch Perceptron Dashboard

สร้าง Django dashboard สำหรับฝึกสอนเพอร์เซปตรอนแบบ real-time (SSE streaming)

## วิธีรัน

```bash
uv sync
uv run manage.py runserver
```

เปิด http://127.0.0.1:8000/

## สิ่งที่นักศึกษาต้องทำ

1. **`dashboard/ml/train.py`** — implement `event_stream()` ให้สมบูรณ์
   (สร้างข้อมูล, perceptron, training loop, บันทึก loss curve)
2. **`dashboard/views.py`** — เปลี่ยน `student_id` และ `student_name` เป็นของจริง

## หมายเหตุ

- Python 3.13 (กำหนดใน `pyproject.toml`)
- ส่งงานผ่าน branch `wk09`
