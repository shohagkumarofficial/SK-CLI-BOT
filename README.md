# 🤖 Advanced Telegram AI Bot

Claude Code CLI-style Telegram bot powered by [AgentRouter.org](https://agentrouter.org/)

---

## ✨ Features

- 🤖 Multiple AI Models (Claude, DeepSeek, GLM)
- 💬 Conversation history with context
- ⚙️ Per-user settings (API key, model, system prompt)
- ➕ Add custom models from within the bot
- 💾 Full database backup & restore system
- 🔒 Secure API key storage (per user)
- 📱 Fully button-based UI

## 🚀 Deploy on Render (Free)

### Step 1 — Create Telegram Bot
1. Open [@BotFather](https://t.me/BotFather) on Telegram
2. Send `/newbot` and follow instructions
3. Copy the **Bot Token**

### Step 2 — Get AgentRouter API Key
1. Visit [agentrouter.org/console/token](https://agentrouter.org/console/token)
2. Sign in and generate an API key (starts with `sk-...`)

### Step 3 — Deploy on Render
1. Push this folder to a **GitHub repository**
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Set environment variable:
   - `TELEGRAM_BOT_TOKEN` = your bot token from BotFather
5. Build command: `pip install -r requirements.txt`
6. Start command: `python bot.py`
7. Add a **Disk** (1GB) mounted at `/opt/render/project/src`
8. Deploy!

### Step 4 — Configure the Bot
1. Start the bot: `/start`
2. Go to ⚙️ Settings → Set API Key
3. Enter your AgentRouter API key
4. Select a model from 🔄 Models
5. Start chatting!

---

## 📋 Bot Commands

| Command | Description |
|---------|-------------|
| `/start` | Main menu |
| `/menu` | Open menu |
| `/clear` | Clear chat history |
| `/backup` | Download DB backup |
| `/restore` | Restore from backup |
| `/cancel` | Cancel current action |

---

## 🤖 Available Models

| Model ID | Name |
|----------|------|
| `claude-haiku-4-5-20251001` | ⚡ Claude Haiku 4.5 |
| `claude-opus-4-6` | 🧠 Claude Opus 4.6 |
| `deepseek-v4-flash` | 🚀 DeepSeek V4 Flash |
| `deepseek-v4-pro` | 💎 DeepSeek V4 Pro |
| `glm-5.1` | 🌟 GLM 5.1 |

You can also **add custom models** from within the bot!

---

## 💾 Backup System

- Click **💾 Backup DB** to download a `.txt` file with your full database
- Click **📤 Restore DB** and send the backup file to restore
- Your API keys, settings, conversation history — all preserved

---

## 🔧 Local Development

```bash
pip install -r requirements.txt
export TELEGRAM_BOT_TOKEN="your-token-here"
python bot.py
```

---

## 📁 Files

```
telegram_ai_bot/
├── bot.py           # Main bot code
├── requirements.txt # Python dependencies
├── render.yaml      # Render deployment config
└── README.md        # This file
```
