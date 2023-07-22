from python:3.11.1-buster

WORKDIR /

RUN pip install runpod torch transformers accelerate bitsandbytes sentencepiece scipy

ADD preload.py .
RUN python preload.py
ADD handler.py .

CMD [ "python", "-u", "/handler.py" ]
