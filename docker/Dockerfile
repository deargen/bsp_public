FROM daeseoklee/bsp-inference:dep2

ENV APP_PATH=/workspace/bsp_public
WORKDIR $APP_PATH
RUN echo "PS1='bsp_public:\[\e[38;5;29;1m\]\W\[\e[38;5;196m\]\\$\[\e[38;5;52m\]:\[\e[0m\]'" >> ~/.bashrc
EXPOSE 5000

COPY ./logs ./logs 
COPY ./scripts ./scripts
COPY ./examples ./examples
COPY ./src ./src

# create directory "/input"
RUN mkdir /input
RUN mkdir /output 

ENV PYTHONPATH $APP_PATH/src
ENTRYPOINT ["/opt/conda/bin/python", "/workspace/bsp_public/src/inference/run.py"]
