FROM huggingface/transformers-pytorch-cpu:latest
COPY ./ /app
WORKDIR /app
RUN pip install -r requirements_inference.txt
EXPOSE 8000
ENV LC_ALL=C.UTF-8 LANG=C.UTF-8
ENV PYTHONPATH "${PYTHONPATH}:./"
CMD ["uvicorn", "--app-dir=src", "web_app:app", "--host", "0.0.0.0", "--port", "8000"]

