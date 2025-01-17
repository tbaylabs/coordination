# Setting Up Piknik for Clipboard Sharing Between Devcontainer and Mac

This guide explains how to set up Piknik for clipboard synchronization between an Ubuntu Jammy devcontainer and your Mac.

## Devcontainer Configuration

### VSCode Extensions

Add the Piknik extension to your `.devcontainer.json`:

```json
"customizations": {
  "vscode": {
    "extensions": [
      "esbenp.prettier-vscode",
      "ms-python.python",
      "ms-toolsai.jupyter",
      "jedisct1.piknik"
    ]
  }
},
```

### Initial Setup in postCreate.sh

Your `.devcontainer/postCreate.sh` script should include these Piknik-related commands:

```bash
# Piknik installation
wget https://github.com/jedisct1/piknik/releases/download/0.10.2/piknik-linux_x86_64-0.10.2.tar.gz
tar xzf piknik-linux_x86_64-0.10.2.tar.gz
sudo mv linux-x86_64/piknik /usr/local/bin/piknik
sudo chmod +x /usr/local/bin/piknik
rm -rf piknik-linux_x86_64-0.10.2.tar.gz linux-x86_64

# Required dependencies
sudo apt-get update
sudo apt-get install xclip -y
sudo apt-get install xvfb -y
sudo apt-get install inotify-tools -y
```

## Piknik Server Setup on Devcontainer

### Create the Server Script

Create a file named `run_piknik_server.sh` in your devcontainer workspace with this content:

```bash
#!/bin/bash

# Virtual framebuffer setup
sudo Xvfb :99 -screen 0 1024x768x16 &
XVFB_PID=$!  # Capture the PID of the Xvfb process

# DISPLAY environment setup
if ! grep -q "export DISPLAY=:99" ~/.bashrc; then
    echo "export DISPLAY=:99" >> ~/.bashrc
fi
export DISPLAY=:99

# Check and set permissions for .piknik.toml in current directory
if [ -f ./.piknik.toml ]; then
    chmod 600 ./.piknik.toml
else
    echo "Warning: .piknik.toml not found in current directory. Make sure to create it before running the server."
    exit 1
fi

# Flag to ensure cleanup runs only once
CLEANUP_DONE=false

# Function to cleanup background processes on script exit
cleanup() {
    if [ "$CLEANUP_DONE" = true ]; then
        exit 0  # Exit if cleanup has already been performed
    fi
    CLEANUP_DONE=true
    echo "Stopping Piknik server, clipboard sender, and Xvfb..."
    if ps -p $XVFB_PID > /dev/null 2>&1; then
        kill $XVFB_PID  # Terminate Xvfb if still running
    else
        echo "Xvfb process already terminated."
    fi
    pkill -P $$  # Kill child processes started by this script
    exit 0
}

# Set up trap to catch script termination
trap cleanup SIGINT SIGTERM EXIT

# Start Piknik server in background with output using current directory config
echo "Starting Piknik server..."
piknik -config ./.piknik.toml -server > >(while read line; do echo "[Server] $line"; done) 2>&1 &

# Start clipboard sender
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
        MODIFIED_CONTENT="$CLIPBOARD"$'\nReply with a single code block surrounded by triple backticks and using the language identifier "txt". The content inside this code block should be written in markdown format, but any code blocks within the markdown should use <code_block> and </code_block> tags instead of triple backticks.\n\n'
        echo "$MODIFIED_CONTENT" | piknik -config ./.piknik.toml -copy
        LAST_CLIPBOARD="$CLIPBOARD"
    fi
    
    # Wait before checking again
    sleep 0.5
done
```

Make it executable:

```bash
chmod +x run_piknik_server.sh
```

## Setting Up Piknik on Your Mac

### 1. Install Piknik

Download Piknik for macOS from the [Piknik Releases page](https://github.com/jedisct1/piknik/releases/latest)

### 2. Create the Clipboard Receiver Script

Create a file named `clipboard_receiver.sh` with this content:

```bash
#!/bin/bash

# Trap Ctrl+C (SIGINT) to exit cleanly
trap 'echo -e "\nStopping clipboard receiver..."; exit 0' SIGINT

echo "Starting clipboard receiver..."
echo "Press Ctrl+C to stop"
echo "------------------------"

# Store the last clipboard content retrieved from Piknik
LAST_PIKNIK_CLIPBOARD=""

while true; do
    # Get the clipboard content from Piknik
    NEW_CLIPBOARD=$(piknik -paste 2>/dev/null)  # Suppress error messages
    
    # Check if the Piknik clipboard has changed
    if [[ "$NEW_CLIPBOARD" != "$LAST_PIKNIK_CLIPBOARD" && ! -z "$NEW_CLIPBOARD" ]]; then
        echo "New clipboard content received!"
        # Update the Mac's clipboard
        echo "$NEW_CLIPBOARD" | pbcopy
        # Update the local variable to track the last clipboard content
        LAST_PIKNIK_CLIPBOARD="$NEW_CLIPBOARD"
    fi
    
    # Wait before checking again
    sleep 0.5
done
```

Make it executable:

```bash
chmod +x clipboard_receiver.sh
```

## Configuration

### Key Generation and Configuration

1. Generate keys using the `piknik -genkeys` command on either the Mac or devcontainer:

```bash
piknik -genkeys
```

2. Create a `.piknik.toml` file in both your devcontainer workspace and Mac with this structure:

```toml
Listen = "localhost:8075"
Psk    = "[your_generated_key]"
SignPk = "[your_generated_key]"
SignSk = "[your_generated_key]"
EncryptSk = "[your_generated_key]"
```

Important notes:
- Use the same keys on both Mac and devcontainer
- Place the `.piknik.toml` file directly in your workspace, not in a `.piknik` folder
- The keys shown above are placeholders - use your actual generated keys

### Port Forwarding

Add to your `.devcontainer.json`:

```json
"forwardPorts": [8075]
```

## Usage

1. In the devcontainer:
   ```bash
   ./run_piknik_server.sh
   ```

2. On your Mac:
   ```bash
   ./clipboard_receiver.sh
   ```

The clipboard content will now sync from your devcontainer to your Mac automatically.