# Run the experiment server in a tmux session

#!/bin/bash

# Usage: ./run_tmux.sh store.js
# Usage: ./run_tmux.sh app.js

FILE=$1

if [ -z "$FILE" ]; then
    echo "Error: No file provided."
    echo "Usage: $0 <node_file.js>"
    exit 1
fi

SESSION="node_$(basename $FILE .js)"
CMD="node $FILE"

# Check if tmux session exists
tmux has-session -t $SESSION 2>/dev/null

if [ $? != 0 ]; then
    echo "Starting new tmux session: $SESSION"
    tmux new-session -d -s $SESSION
    tmux send-keys -t $SESSION "$CMD" C-m
else
    echo "Session '$SESSION' already exists. Attaching..."
fi

# Attach to the session
tmux attach -t $SESSION