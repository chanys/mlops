FROM huggingface/transformers-pytorch-cpu:latest
COPY ./ /app
WORKDIR /app

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

# aws credentials configuration
ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY

RUN pip install "dvc[s3]"
RUN pip install -r requirements_inference.txt

# to initialize DVC in a directory that is not part of a Git repo
RUN dvc init --no-scm -f

RUN dvc remote add -d storage s3://mlops-demo-dvc/model_output
RUN cat .dvc/config

# pulling the trained model
RUN dvc pull expts/dvcfiles/model.onnx.dvc -f

EXPOSE 8000
ENV LC_ALL=C.UTF-8 LANG=C.UTF-8
ENV PYTHONPATH "${PYTHONPATH}:./"
CMD ["uvicorn", "--app-dir=src", "web_app:app", "--host", "0.0.0.0", "--port", "8000"]

