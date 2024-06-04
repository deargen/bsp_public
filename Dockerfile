FROM daeseoklee/bsp_public:dep2

ENV APP_PATH=/workspace/bsp_public
WORKDIR $APP_PATH
COPY . . 
ENV PYTHONPATH=$APP_PATH/src
RUN echo "PS1='bsp_public:\[\e[38;5;29;1m\]\W\[\e[38;5;196m\]\\$\[\e[38;5;52m\]:\[\e[0m\]'" >> ~/.bashrc
EXPOSE 5000
