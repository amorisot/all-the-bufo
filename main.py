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


@app.get("/")
async def root():
    return HTMLResponse(content=index_html, status_code=200)


@app.post("/get_images/")
async def get_images(query: str = Form(None)):
    print(query)
    # in case of empty query, return a shrug bufo
    if not query:
        name = "bufo-shrug.png"
        image_path = f'all-the-bufo/{name}'
        image = Image.open(image_path)
        image_byte_arr = BytesIO()
        image.save(image_byte_arr, format=image.format)
        image_byte_arr = image_byte_arr.getvalue()
        image_base64 = str(base64.b64encode(image_byte_arr), 'utf-8')
        return {"images": [image_base64], "names": [name]}
    search_embedding = co.embed(texts=[query], model='embed-english-v3.0', input_type='search_query').embeddings[0]
    scores = np.dot(embeddings, np.array(search_embedding))

    images, names = [], []
    for i in np.argsort(scores)[::-1][:8]:
        if bufos[i].endswith(".gif"):
            # deal with gifs
            gif_path = f'all-the-bufo/{bufos[i]}'
            with open(gif_path, "rb") as gif_file:
                gif_data = gif_file.read()
                gif_base64 = str(base64.b64encode(gif_data), 'utf-8')
            images.append(gif_base64)
            names.append(bufos[i])
        else:    
            image_path = f'all-the-bufo/{bufos[i]}'
            image = Image.open(image_path)
            image_byte_arr = BytesIO()
            image.save(image_byte_arr, format=image.format)
            image_byte_arr = image_byte_arr.getvalue()
            image_base64 = str(base64.b64encode(image_byte_arr), 'utf-8')
            images.append(image_base64)
            names.append(bufos[i])

    return {"images": images, "names": names}


if __name__ == "__main__":
    uvicorn.run(app, host="100.69.202.22", port=8000)
    ## to run locally:
    # uvicorn.run(app, host="localhost", port=8000)
