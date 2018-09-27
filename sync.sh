#!/bin/bash

DIR=`dirname $0`
DEST=brdw0:gdasync

rsync -rvh $DIR/workspace/ ${DEST}/workspace/
rsync -rvh $DIR/Libraries/libmp/src/ ${DEST}/Libraries/libmp/src/
rsync -rvh $DIR/Libraries/libmp/include/ ${DEST}/Libraries/libmp/include/
rsync -rvh $DIR/Libraries/libgdsync/src/ ${DEST}/Libraries/libgdsync/src/
rsync -rvh $DIR/Libraries/libgdsync/include/ ${DEST}/Libraries/libgdsync/include/
#rsync -rvh $DIR/Scripts/ ${DEST}/Scripts/
