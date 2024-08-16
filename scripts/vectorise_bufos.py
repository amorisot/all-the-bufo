import os

import cohere
import numpy as np

def main():
    # grab all the filenames in the all-the-bufo directory
    bufos = []
    for filename in os.listdir('all-the-bufo'):
        with open(f'all-the-bufo/{filename}', 'rb') as f:
            bufos.append((f'all-the-bufo/{filename}', f.read()))

    co = cohere.Client(os.environ['COHERE_API_KEY'])
    embeddings = np.array(co.embed(texts=[b[0] for b in bufos], model='embed-english-v3.0', input_type='search_document').embeddings)
    np.save('embufo.npy', embeddings)

if __name__ == '__main__':
    main()
