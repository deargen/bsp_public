FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# Basic setup
ENV APP_PATH=/workspace/bsp_public
WORKDIR $APP_PATH
COPY ./docker_requirements.txt . 

# Install python dependencies
RUN apt-get update
RUN pip install -r docker_requirements.txt

# Install Fpocket 
RUN apt-get install -y make gcc g++ git
WORKDIR /install
RUN git clone https://github.com/Discngine/fpocket.git
WORKDIR /install/fpocket
RUN make && make install
WORKDIR $APP_PATH

# Install other tools
RUN apt-get install -y nano


# RUN conda install -y -c conda-forge openbabel