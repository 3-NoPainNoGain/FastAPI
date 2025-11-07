import uvicorn

if __name__ == "__main__":
    print("FastAPI 서버를 시작합니다...")
    print("http://127.0.0.1:8000")
    
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)