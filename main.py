import os
from io import BytesIO

import base64
import cohere
import uvicorn
import numpy as np

from PIL import Image
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

co = cohere.Client(os.environ['COHERE_API_KEY'])
app = FastAPI()
bufos = sorted(os.listdir('all-the-bufo'))
embeddings = np.load(open('embufo.npy', 'rb'), allow_pickle=True)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
with open("index.html") as f:
    index_html = f.read()


# just display index.html, located in the same directory as main.py
@app.get("/")
async def root():
    return HTMLResponse(content=index_html, status_code=200)


@app.post("/get_images/")
async def get_images(query: str = Form(...)):
    print(query)
    search_embedding = co.embed(texts=[query], model='embed-english-v3.0', input_type='search_query').embeddings[0]
    scores = np.dot(embeddings, np.array(search_embedding))

    images = []
    for i in np.argsort(scores)[::-1][:8]:
        image_path = f'all-the-bufo/{bufos[i]}'
        image = Image.open(image_path)
        image_byte_arr = BytesIO()
        image.save(image_byte_arr, format=image.format)
        image_byte_arr = image_byte_arr.getvalue()
        image_base64 = str(base64.b64encode(image_byte_arr), 'utf-8')
        images.append(image_base64)

    return {"images": images}

@app.post("/get_images_rerank/")
async def get_images(query: str = Form(...)):
    print(query)
    search_embedding = co.embed(texts=[query], model='embed-english-v3.0', input_type='search_query').embeddings[0]
    scores = np.dot(embeddings, np.array(search_embedding))

    sorted_scores = np.argsort(scores)[::-1][:1000]
    sorted_bufos = [bufos[i] for i in sorted_scores]

    reranked = co.rerank(query=query, documents=sorted_bufos, model='rerank-english-v3.0', top_n=8).results
    reranked_indices = [r.index for r in reranked]

    images = []
    for i in reranked_indices:
        image_path = f'all-the-bufo/{bufos[i]}'
        image = Image.open(image_path)
        image_byte_arr = BytesIO()
        image.save(image_byte_arr, format=image.format)
        image_byte_arr = image_byte_arr.getvalue()
        image_base64 = str(base64.b64encode(image_byte_arr), 'utf-8')
        images.append(image_base64)

    return {"images": images}

if __name__ == "__main__":
    uvicorn.run(app, host="100.69.202.22", port=8000)
