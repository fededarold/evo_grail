#FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime

FROM ubuntu:20.04
ENV DEBIAN_FRONTEND noninteractive
#RUN apt-get update && apt-get install -y python3.10 python3.10-dev
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update 
RUN apt-get install -y python3.10
RUN apt-get install -y python3-pip 
RUN apt-get install --no-install-recommends  
RUN unzip 
RUN apt-get install -y libglu1-mesa-dev 
RUN apt-get install -y libgl1-mesa-dev 
RUN apt-get install -y libosmesa6-dev 
RUN apt-get install xvfb 
RUN apt-get install patchelf 
RUN apt-get install -y ffmpeg cmake
RUN apt-get install -y swig
RUN apt-get install wget 
RUN apt-get autoremove 
RUN apt-get clean 
RUN rm -rf /var/lib/apt/lists/* 
    # Download mujoco
RUN mkdir /root/.mujoco 
RUN cd /root/.mujoco 
RUN wget -qO- 'https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz' | tar -xzvf -

ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin"

COPY . /usr/local/gym/
WORKDIR /workspace/

#RUN if [ "python:${PYTHON_VERSION}" = "python:3.6.15" ] ; then pip install .[box2d,classic_control,toy_text,other] pytest=="7.0.1" --no-cache-dir; else pip install .[testing] --no-cache-dir; fi
RUN pip install gymnasium[all]
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install scipy
RUN pip install numpy
RUN pip install matplotlib
RUN pip install PyYAML

COPY ./ /workspace/
CMD ["python3","-u","/workspace/main.py"]
#FROM continuumio/miniconda3

# SET BASH AS CURRENT SHELL
#RUN chsh -s /bin/bash
#SHELL ["/bin/bash", "-c"]

#COPY ./pillar.yml /tmp/pillar.yml

#RUN conda update conda \
#    && conda env create -f /tmp/pillar.yml

#RUN echo "conda activate nib_docker" >> ~/.bashrc
#ENV PATH /opt/conda/envs/nib_docker/bin:$PATH
#ENV CONDA_DEFAULT_ENV $nib_docker

#COPY . .
#CMD [ "python", "-u", "train.py" ]
