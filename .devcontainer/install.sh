#!/bin/bash

DEV_PATH="/workspaces/$(basename $(pwd))/.devcontainer"
MODEL_FILE="/workspaces/$(basename $(pwd))/Qwen3-0.6B-Q8_0.gguf"

# Check if model file already exists
if [ -f "$MODEL_FILE" ]; then
    echo "✅ Model already exists, skipping setup."
    exit 0
fi

# Optionally remove copilot
# rm -rf ~/.vscode-remote/extensions/github.copilot* && chmod -w ~/.vscode-remote/extensions

# Ensure devcontainer path exists
mkdir -p "$DEV_PATH"

# Set environment variables if not already in .bashrc
BASHRC="$HOME/.bashrc"
grep -qxF "export LLAMA_CPP_LIB_PATH=$DEV_PATH" "$BASHRC" || echo "export LLAMA_CPP_LIB_PATH=$DEV_PATH" >> "$BASHRC"
grep -qxF "export LD_LIBRARY_PATH=$DEV_PATH:\$LD_LIBRARY_PATH" "$BASHRC" || echo "export LD_LIBRARY_PATH=$DEV_PATH:\$LD_LIBRARY_PATH" >> "$BASHRC"

export LLAMA_CPP_LIB_PATH=$DEV_PATH
export LLAMA_CPP_LIB="$DEV_PATH/libllama.so"
export LD_LIBRARY_PATH=$DEV_PATH:$LD_LIBRARY_PATH

# Install Python package
echo "⏳ Installing inference engine (llama-cpp-python)..."
export CMAKE_ARGS="-DLLAMA_BUILD=OFF"
pip install -q --upgrade pip
pip install -q llama-cpp-python==0.3.10
echo "✅ Inference engine installed."

# Download model
echo "⏳ Downloading model..."
wget -q -O "$MODEL_FILE" https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf
echo "✅ Model downloaded."

echo "⏳ Installing huggingface libraries..."
pip install -q torch --index-url https://download.pytorch.org/whl/cpu
pip install -q sentence-transformers faiss-cpu

echo ""
echo "✅ DevContainer setup complete! You can now start working on your assignment."
