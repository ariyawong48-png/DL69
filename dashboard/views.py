import json
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from django.shortcuts import render
from django.http import StreamingHttpResponse

# --- 1. โครงสร้างโมเดล ---
class SimpleRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 1)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

class SimpleLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# --- 2. Views หลัก ---
def wk11_train(request):
    return render(request, 'dashboard/wk11_train.html')

def wk11_load(request):
    # รับค่าชนิดโมเดลจาก URL parameter (ถ้าไม่มีให้ใช้ lstm เป็นค่าเริ่มต้น)
    model_type = request.GET.get('model', 'lstm')
    file_name = f'{model_type}_model.pt'

    try:
        # 1. อ่านข้อมูล Jena Climate
        df = pd.read_csv('jena_climate_2009_2016.csv')
        temp = df['T (degC)'].values.astype(np.float32)
        
        scaler = MinMaxScaler()
        temp_scaled = scaler.fit_transform(temp.reshape(-1, 1)).flatten()
        
        LOOKBACK = 60
        X_test, y_test = [], []
        for i in range(100):
            X_test.append(temp_scaled[i : i + LOOKBACK])
            y_test.append(temp_scaled[i + LOOKBACK])
            
        x_tensor = torch.tensor(np.array(X_test), dtype=torch.float32).unsqueeze(-1)
        
        # 2. โหลดโมเดลตามประเภทที่เลือก (RNN หรือ LSTM)
        model = SimpleRNN() if model_type == 'rnn' else SimpleLSTM()
        model.load_state_dict(torch.load(file_name))
        model.eval()
        
        # 3. ทำนายผล
        with torch.no_grad():
            preds_scaled = model(x_tensor).numpy().flatten()
            
        # 4. แปลงค่ากลับเป็นองศาจริง
        actual = scaler.inverse_transform(np.array(y_test).reshape(-1, 1)).flatten()
        preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        
        # 5. คำนวณค่า MAE
        mae = round(float(np.mean(np.abs(actual - preds))), 4)
        
        context = {
            'status': f'โหลดโมเดล {file_name} สำเร็จ พร้อมใช้งานทำนายผล!',
            'mae': mae,
            'selected_model': model_type,
            'actual_data': json.dumps(actual.tolist()[:30]),
            'pred_data': json.dumps(preds.tolist()[:30]),
            'labels': json.dumps(list(range(1, 31)))
        }
    except Exception as e:
        context = {
            'status': f'เกิดข้อผิดพลาดในการโหลด {file_name}: กรุณาเทรนโมเดลชนิดนี้ก่อน ({str(e)})',
            'mae': 'N/A',
            'selected_model': model_type,
            'actual_data': '[]',
            'pred_data': '[]',
            'labels': '[]'
        }
        
    return render(request, 'dashboard/wk11_load.html', context)

# --- 3. SSE Stream อ่านข้อมูล Jena จริง ---
def wk11_train_stream(request):
    model_type = request.GET.get('model', 'rnn')
    
    def event_stream():
        # โหลดข้อมูล Jena Climate และดึงคอลัมน์อุณหภูมิ T (degC)
        df = pd.read_csv('jena_climate_2009_2016.csv')
        temp = df['T (degC)'].values.astype(np.float32)
        
        # Normalize ข้อมูล
        scaler = MinMaxScaler()
        temp_scaled = scaler.fit_transform(temp.reshape(-1, 1)).flatten()
        
        # เตรียม Windowing (lookback = 60 ตามโจทย์)
        LOOKBACK = 60
        # สุ่มสุ่มตัวอย่าง 1000 จุดเพื่อความรวดเร็วในการเทรนสตรีมมิงบนเว็บ
        X_list, y_list = [], []
        for i in range(1000):
            X_list.append(temp_scaled[i : i + LOOKBACK])
            y_list.append(temp_scaled[i + LOOKBACK])
            
        x_data = torch.tensor(np.array(X_list), dtype=torch.float32).unsqueeze(-1) # Shape: (1000, 60, 1)
        y_data = torch.tensor(np.array(y_list), dtype=torch.float32).unsqueeze(-1) # Shape: (1000, 1)

        # เลือกโมเดล
        model = SimpleRNN() if model_type == 'rnn' else SimpleLSTM()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # lr=0.001 ตามโจทย์
        criterion = nn.MSELoss()

        for epoch in range(1, 21): # 20 Epochs ตามโจทย์
            optimizer.zero_grad()
            output = model(x_data)
            loss = criterion(output, y_data)
            loss.backward()
            optimizer.step()

            data = json.dumps({'epoch': epoch, 'loss': round(loss.item(), 5)})
            yield f"data: {data}\n\n"
            time.sleep(0.1)

        # บันทึก model ด้วย state_dict() ตามโจทย์ข้อ 6
        torch.save(model.state_dict(), f'{model_type}_model.pt')
        yield f"data: {json.dumps({'status': 'complete'})}\n\n"

    response = StreamingHttpResponse(event_stream(), content_type='text/event-stream')
    response['Cache-Control'] = 'no-cache'
    return response

# def wk11_load(request):
#     # สร้างโครงสร้างโมเดลเปล่า และโหลดน้ำหนักที่เซฟไว้เข้ามา
#     model = SimpleLSTM()
#     try:
#         model.load_state_dict(torch.load('lstm_model.pt'))
#         model.eval()
#         status = "โหลดโมเดล lstm_model.pt สำเร็จ พร้อมใช้งานทำนายผล!"
#     except Exception as e:
#         status = f"ยังไม่พบไฟล์โมเดลที่เทรนเสร็จ (กรุณาเทรนที่หน้า /train/ ก่อน): {str(e)}"

#     return render(request, 'dashboard/wk11_load.html', {'status': status})