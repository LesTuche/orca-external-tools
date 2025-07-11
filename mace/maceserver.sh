#!/bin/bash

# Run the maceserver script with all passed arguments.
# Assumes the virtual environment with maceexttool is already activated
maceserver "$@" &
PID=$!
echo "MACESERVER_PID: $PID"
