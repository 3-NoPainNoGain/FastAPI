@app.get("/")
def root():
    return {"message": "Handoc Backend API"}

@app.get("/health")
def health():
    return {"status": "ok"}