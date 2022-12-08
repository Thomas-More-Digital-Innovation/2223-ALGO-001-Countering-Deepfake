FROM python:3.10

WORKDIR /deepfake-project

COPY main.py .
COPY data_prep.py .
COPY requirements.txt .
COPY deepfake_model.h5 .

RUN mkdir -p ./imagery/boxed_frames
RUN mkdir -p ./imagery/faces
RUN mkdir -p ./imagery/frames
RUN mkdir -p ./imagery/video
RUN mkdir -p ./imagery/zipped

COPY imagery/arrow-64.png ./imagery/

RUN apt-get update && apt-get -y install cmake
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r requirements.txt
RUN pip install tensorflow-cpu

CMD ["streamlit", "run", "./main.py"]