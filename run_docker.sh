#!/bin/bash
## This script will use the default cmd in the Dockerfile to run jupyter server
## The server will start on localhost port 8888
## Use the link provided in the terminal to access the server via browser, or copy the link for ipykernel selection in VSCode

source image_description.conf
docker run --rm --gpus all --ipc=host -v $(pwd)/notebooks:/mnt -p 8888:8888 --name "RL_course_jupyter_server" $image_name:$image_tag