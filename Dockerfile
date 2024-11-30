FROM ubuntu:20.04
# pytorch image wont cut it...
USER root
ENV HOME_DIR /home/rl_course
WORKDIR $HOME_DIR
RUN mkdir -p $HOME_DIR
ENV MOUNT_DIR /mnt
RUN mkdir -p $MOUNT_DIR

# System level dependencies
RUN apt-get update 
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y swig cmake ffmpeg freeglut3-dev xvfb git wget unzip tldr-py vim
# python3.7 is specifically needed for rl cardiac
RUN apt-get install -y python3.7 python3.7-dev python3.7-distutils python3-pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip install --upgrade pip

# RL course dependencies
## using pytorch docker
# RUN pip install torch==2.3.1 
RUN pip install "autorom[accept-rom-license]"
RUN pip install opencv-python-headless
RUN pip install --ignore-installed rl-zoo3==2.0.0
# to install the latest commit of a repo
ADD "https://api.github.com/repos/yusenz/gym-maze/commits?per_page=1" latest_commit
RUN git clone https://github.com/yusenz/gym-maze.git $HOME_DIR/gym-maze
RUN cd $HOME_DIR/gym-maze && python setup.py install
# RUN git clone https://github.com/DLR-RM/rl-baselines3-zoo $HOME_DIR/rl-baselines3-zoo
# RUN cd $HOME_DIR/rl-baselines3-zoo && pip install -r requirements.txt
# RUN pip install -e .[plots,tests]
## rl cardiac
RUN pip install scikit-learn==0.23.2
RUN pip install tensorflow[and-cuda]==2.9.1
COPY ./rl_cardiac $HOME_DIR/rl_cardiac
## RL-DBS
COPY ./rl_dbs $HOME_DIR/rl_dbs
# TVB
RUN pip install tvb-library
RUN pip install tvb-framework
COPY ./TVB $HOME_DIR/TVB

# Jupyter
RUN pip install jupyter
EXPOSE 8888
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]

# some sanity python dependencies, put last so that only installed if not exist
RUN pip install numpy scipy matplotlib pandas pygame
RUN cd $HOME_DIR