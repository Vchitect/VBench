#!/bin/bash
CACHE_DIR=~/.cache/vbench
wget -P $CACHE_DIR/raft_model/ https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip
unzip -d ${CACHE_DIR}/raft_model/ $CACHE_DIR/raft_model/models.zip
rm -r $CACHE_DIR/raft_model/models.zip
