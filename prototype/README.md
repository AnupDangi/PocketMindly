# PocketMindly - Voice Assistant

## Quick Start

```bash
cd prototype
./run.sh
```

## Features

- ✅ **Auto speech detection** - No button pressing needed
- ✅ **Smart silence detection** - 1.5s pause to end recording
- ✅ **Web search** - Automatic when LLM needs info
- ✅ **Natural conversation** - Continuous listening

## How It Works

1. **Speak naturally** - System detects when you start
2. **Pause 1.5 seconds** - System knows you're done
3. **AI responds** - With web search if needed
4. **Repeat** - Always listening

## Test It

Try these:
- "What is the capital of France?" (general knowledge)
- "Who is Elon Musk?" (web search)
- "Tell me about space" (conversation)

## Fixed Issues

- ✅ Increased silence threshold to 1.5s (was cutting off speech)
- ✅ Fixed web search response (was showing SEARCH command instead of answer)
- ✅ Disabled noisy partial transcripts (final transcript is cleaner)

## Performance

- **Silence detection**: 1.5s
- **Total latency**: ~1.5-2s
- **Feels natural**: Yes!
