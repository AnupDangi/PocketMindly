#!/bin/bash
# PocketMindly - Voice Assistant Launcher

cd "$(dirname "$0")"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "╔════════════════════════════════════════════════════════════╗"
echo "║           PocketMindly - AI Voice Assistant               ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Features:"
echo "  ✓ Real-time speech detection (no button pressing)"
echo "  ✓ Live transcription (see what you're saying)"
echo "  ✓ Web search integration (LLM-driven)"
echo "  ✓ Natural conversation flow"
echo ""
echo "Starting..."
echo ""

python3 main.py
