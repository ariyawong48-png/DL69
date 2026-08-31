"""
การบ้านสัปดาห์ที่ 9 — PyTorch Perceptron Dashboard

นักศึกษาต้องเขียนโค้ดในไฟล์นี้ให้สมบูรณ์ โดยต้องมีฟังก์ชัน:

    def event_stream():
        # generator ที่ yield dict ข้อมูลทีละ epoch แล้ว stream ผ่าน SSE

yield dict ที่มี key ต่อไปนี้ (แต่ละ epoch ระหว่างฝึกสอน):
    {
        "type": "progress",
        "epoch": <int>,
        "loss": <float>,
        "accuracy": <float>,
    }

และเมื่อฝึกสอนเสร็จ yield dict สุดท้าย:
    {
        "type": "done",
        "results": <ผลสรุป เช่น ลิงก์ไปยัง loss_curve.png>,
    }

ข้อกำหนด:
    1. สร้างข้อมูล binary classification (linear separable) ด้วย torch.randn
    2. สร้าง perceptron = nn.Linear(2, 1) + nn.Sigmoid()
    3. Training loop ด้วยมือ: forward -> loss -> backward -> update weights
    4. ทดลอง learning rate 3 ค่า: 0.001, 0.1, 1.0
    5. บันทึก loss curve เป็น PNG ที่ dashboard/static/dashboard/loss_curve.png
        แล้ว return เส้นทางให้เบราว์เซอร์แสดงผล (เช่น "static/dashboard/loss_curve.png")

ดูตัวอย่างโค้ดได้จาก slides wk09.html และ wk09-hw.py
"""

# TODO: เขียนโค้ดการบ้านตรงนี้
import os
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def event_stream():
    """
    generator สำหรับ stream ผลการฝึกสอนผ่าน SSE

    ตัวอย่างการใช้งานใน views.train_stream():
        for event in train.event_stream():
            yield ...
    """    # ----- ตัวอย่างโครงสร้าง (นักศึกษาต้องเขียนให้สมบูรณ์) -----

    # 1. สร้างข้อมูล (linear separable binary classification)
    #    X = torch.randn(200, 2)
    #    y = ((X[:, 0] * 1.5 + X[:, 1] - 0.5) > 0).float().unsqueeze(1)
    torch.manual_seed(42)
    X = torch.randn(200, 2)
    y = ((X[:, 0] * 1.5 + X[:, 1] - 0.5) > 0).float().unsqueeze(1)

    # 2. สร้างโมเดล perceptron
    model = nn.Sequential(
        nn.Linear(2, 1),
        nn.Sigmoid(),
    )

    loss_fn = nn.BCELoss()

    # ลอง learning rates ต่าง ๆ
    learning_rates = [0.001, 0.1, 1.0]
    num_epochs = 100
    all_losses = {}

    # กำหนด static_dir สำหรับบันทึกไฟล์กราฟ PNG
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    static_dir = os.path.join(BASE_DIR, 'static', 'dashboard')
    os.makedirs(static_dir, exist_ok=True)

    for lr in learning_rates:
        # reset weights ใหม่สำหรับแต่ละ lr
        for layer in model:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

        losses = []

        for epoch in range(1, num_epochs + 1):
            # 3. training loop ด้วยมือ
            y_hat = model(X)
            loss = loss_fn(y_hat, y)

            loss.backward()

            with torch.no_grad():
                for p in model.parameters():
                    p -= lr * p.grad
                model.zero_grad()

            # คำนวณ accuracy
            acc = float(((y_hat > 0.5).float() == y).float().mean().item())
            losses.append(loss.item())

            # ส่งค่า Real-time ผ่าน SSE ทุก epoch สำหรับ lr = 0.1
            if lr == 0.1:
                yield {
                    "type": "progress",
                    "epoch": epoch,
                    "loss": loss.item(),
                    "accuracy": acc,
                }

        all_losses[lr] = losses

    # 4. บันทึก loss curve (สร้างและเซฟไฟล์ PNG จริงลงโฟลเดอร์ static)
    plt.figure(figsize=(8, 5))
    for lr, losses in all_losses.items():
        plt.plot(range(1, num_epochs + 1), losses, label=f'lr={lr}')
    plt.title('Loss Curve Comparison for 3 Learning Rates')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(static_dir, 'loss_curve.png')
    plt.savefig(save_path)
    plt.close()

    # yield สรุปผลลัพธ์สุดท้าย
    yield {
        "type": "done",
        "results": "static/dashboard/loss_curve.png",  # สร้างไฟล์จริงเรียบร้อย
    }