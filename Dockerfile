name: Deploy METAL backend with local llama.cpp running Mistral

on:
  push:
    branches:
      - main
  workflow_dispatch:  

env:
  IMAGE_NAME: michaelsigamani/naturalplan-benchmark
  IMAGE_TAG: latest
  HF_API_KEY: ${{ secrets.HF_API_KEY }}
  MODEL_NAME: "bartowski/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf" 
  EMBEDDING_MODEL: "nomic-ai/nomic-embed-text-v1.5-GGUF"
  
jobs:
  build-and-deploy:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout repository
      uses: actions/checkout@v4

    - name: Log in to Docker Hub
      uses: docker/login-action@v3
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install Dependencies
      run: |
        pip install --no-cache-dir --upgrade pip
        pip install -r requirements.txt

    - name: Pre-download Toy Model (For Testing )
      run: |
        mkdir -p $HOME/models/huggingface
        HF_HOME=$HOME/models/huggingface python3 -c \
          "from transformers import AutoModel; AutoModel.from_pretrained('distilbert-base-uncased', cache_dir='$HOME/models/huggingface')"
 
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
             
    - name: Build and push Docker image
      run: |
        docker buildx build --push \
                            --build-arg HF_API_KEY=${{ secrets.HF_API_KEY }} \
                            --build-arg MODEL_NAME="NousResearch/Meta-Llama-3-8B-Instruct" \
                            --build-arg EMBEDDING_MODEL="nomic-ai/nomic-embed-text-v1.5-GGUF" \
                            -t $IMAGE_NAME:$IMAGE_TAG \
                            .
