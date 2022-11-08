#!/bin/bash
set -e # exit when a command fails
set -u # exit when trying to use undeclared variables

unzip FB15K-237-modified.zip
cp -r Release/* .
rm -rf Release
