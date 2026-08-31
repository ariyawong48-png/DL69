import json

from django.http import StreamingHttpResponse
from django.shortcuts import render

from .ml import train


def index(request):
    """Landing page — ใส่รหัสนักศึกษาและชื่อที่นี่ (ในเอกสารการบ้าน)"""
    context = {
        "student_id": "67114540639",       # ★ แก้ไขเป็นรหัสนักศึกษาจริง
        "student_name": "อริยวงศ์ วิสัยการ",  # ★ แก้ไขเป็นชื่อ-นามสกุลจริง
    }
    return render(request, "dashboard/index.html", context)


def train_stream(request):
    """SSE: สตรีมผลการฝึกสอนแบบ real-time ไปยังเบราว์เซอร์

    เรียก ml.train.event_stream() ซึ่งเป็น generator ที่ yield ข้อมูล
    เป็น JSON ทีละ epoch แล้ว stream ผ่าน StreamingHttpResponse
    """

    def event_stream():
        for event in train.event_stream():
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingHttpResponse(event_stream(), content_type="text/event-stream")