from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from data_loader import load_data
from preprocessing import preprocess_data
from model import LSTMModel

app = FastAPI(
    title="Stock Price Prediction API",
    description="API для прогнозирования цен на акции с использованием модели LSTM.",
    version="1.0.0"
)

class PredictionRequest(BaseModel):
    ticker: str
    start_date: str
    end_date: str

# Загрузка модели
try:
    model = LSTMModel()
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {e}")
    raise

@app.post("/predict/", summary="Получить предсказание цены на акцию")
async def predict(request: PredictionRequest):
    """
    Получить предсказание цены на акцию для заданного тикера и временного периода.
    
    - **ticker**: Тикер компании (например, AAPL).
    - **start_date**: Начальная дата в формате YYYY-MM-DD.
    - **end_date**: Конечная дата в формате YYYY-MM-DD.
    
    Возвращает предсказанную цену на акцию.
    """
    try:
        # Загрузка и предобработка данных
        data = load_data(request.ticker, request.start_date, request.end_date)
        seq_length = 60
        X, y, scaler = preprocess_data(data, seq_length)
        if len(X) < seq_length:
            raise ValueError(f"Not enough data for ticker {request.ticker} to create sequences of length {seq_length}.")
        X_test = X[-1:]  # Используем последнюю последовательность для предсказания
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(2)
        
        # Предсказание
        with torch.no_grad():
            prediction = model(X_test_tensor)
        
        # Обратное преобразование предсказания
        prediction_scaled = scaler.inverse_transform(prediction.numpy().reshape(-1, 1))
        return {"prediction": prediction_scaled.tolist()[0][0]}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)