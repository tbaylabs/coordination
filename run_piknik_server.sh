#!/bin/bash
export DISPLAY=:99

# Function to cleanup background processes on script exit
cleanup() {
    echo "Stopping Piknik server and clipboard sender..."
    pkill -P $  # Kill all child processes
    exit 0
}

# Set up trap to catch script termination
trap cleanup SIGINT SIGTERM

# Start Piknik server in background with output
echo "Starting Piknik server..."
piknik -server > >(while read line; do echo "[Server] $line"; done) 2>&1 &

echo "Starting clipboard sender..."
# Previous clipboard content
LAST_CLIPBOARD=""

# Clipboard sender loop
while true; do
    # Get current clipboard content
    CLIPBOARD=$(xclip -o -selection clipboard 2>/dev/null || echo "")
    
    # Check if the clipboard content has changed
    if [[ "$CLIPBOARD" != "$LAST_CLIPBOARD" ]]; then
        echo "[Sender] Copying new content to Piknik"
        # Add the clipboard content followed by the instruction line
MODIFIED_CONTENT="$CLIPBOARD"$'\nReply with a single code block surrounded by triple backticks and using the language identifier "txt". The content inside this code block should be written in markdown format, but any code blocks within the markdown should use <code_block> and </code_block> tags instead of triple backticks.\n\n'        echo "$MODIFIED_CONTENT" | piknik -copy
        LAST_CLIPBOARD="$CLIPBOARD"
    fi
    
    # Wait before checking again
    sleep 0.5
done