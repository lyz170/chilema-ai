### 0. setup langchain
```bash
pip install -U langchain
pip install -U langgraph
pip install -U langchain-deepseek
pip install python-dotenv
```

### 1. install following packages
```bash
pip install fastapi uvicorn
```

### 2. run the server
```bash
cd chatbot
uvicorn main:app --reload --port 8000
```

### 3. test the server
```bash
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json; charset=utf-8"  -d '{"message":"3+5=?","history":[]}'
```