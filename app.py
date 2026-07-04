from main import graph
from fastapi import FastAPI


app =FastAPI()

@app.get("/")
def reed_root():
    return {"message": "Hello, World!"}

@app.post("/summarizer")
def video_summarizer(video_url: str):
    result = graph.invoke(input={"video_url": video_url})
    return result