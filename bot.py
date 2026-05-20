"""
Advanced Telegram AI Bot - Claude Code CLI Style
Powered by AgentRouter.org API
"""

import os
import json
import logging
import asyncio
import sqlite3
import base64
import io
import threading
from datetime import datetime
from typing import Optional
from http.server import HTTPServer, BaseHTTPRequestHandler

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    BotCommand,
)
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ConversationHandler,
    ContextTypes,
    filters,
)
from telegram.constants import ParseMode, ChatAction

import httpx

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
AGENTROUTER_BASE_URL = "https://agentrouter.org"
ATXP_BASE_URL = "https://llm.atxp.ai/v1"
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
DB_PATH = "bot_data.db"

# Provider names
PROVIDER_AGENTROUTER = "agentrouter"
PROVIDER_ATXP = "atxp"
PROVIDER_FIREWORKS = "fireworks"

AGENTROUTER_MODELS = {
    "claude-haiku-4-5-20251001": "⚡ Claude Haiku 4.5",
    "claude-opus-4-6": "🧠 Claude Opus 4.6",
    "deepseek-v4-flash": "🚀 DeepSeek V4 Flash",
    "deepseek-v4-pro": "💎 DeepSeek V4 Pro",
    "glm-5.1": "🌟 GLM 5.1",
}

# Short key → full model ID mapping (Telegram callback_data max 64 bytes)
FIREWORKS_MODEL_MAP = {
    "fw1": "accounts/fireworks/models/kimi-k2-instruct",
    "fw2": "accounts/fireworks/models/deepseek-v3-0324",
    "fw3": "accounts/fireworks/models/glm-4-32b-0414",
    "fw4": "accounts/fireworks/models/qwen3-coder-480b-a35b-instruct",
    "fw5": "accounts/fireworks/models/deepseek-v3",
    "fw6": "accounts/fireworks/models/minimax-m2-0519",
    "fw7": "accounts/fireworks/models/llama4-maverick-instruct-basic",
    "fw8": "accounts/fireworks/models/qwen3-30b-a3b",
    "fw9": "accounts/fireworks/models/llama-v3p3-70b-instruct",
    "fw10": "accounts/fireworks/models/deepseek-r1",
}
# Reverse map: full ID → short key
FIREWORKS_MODEL_RMAP = {v: k for k, v in FIREWORKS_MODEL_MAP.items()}

FIREWORKS_MODELS = {
    "accounts/fireworks/models/kimi-k2-instruct": "🥇 Kimi K2.6 (Best Coding)",
    "accounts/fireworks/models/deepseek-v3-0324": "🥈 DeepSeek V4 Pro",
    "accounts/fireworks/models/glm-4-32b-0414": "🥉 GLM-5.1",
    "accounts/fireworks/models/qwen3-coder-480b-a35b-instruct": "💻 Qwen3 Coder 480B",
    "accounts/fireworks/models/deepseek-v3": "⚡ DeepSeek V3.2",
    "accounts/fireworks/models/minimax-m2-0519": "💡 MiniMax M2.5",
    "accounts/fireworks/models/llama4-maverick-instruct-basic": "🦙 Llama 4 Maverick",
    "accounts/fireworks/models/qwen3-30b-a3b": "🚀 Qwen3 30B",
    "accounts/fireworks/models/llama-v3p3-70b-instruct": "🐪 Llama 3.3 70B",
    "accounts/fireworks/models/deepseek-r1": "🧠 DeepSeek R1",
}

ATXP_MODELS = {
    "claude-opus-4-7": "🧠 Claude Opus 4.7",
    "claude-opus-4-6": "🔮 Claude Opus 4.6",
    "claude-sonnet-4-6": "✨ Claude Sonnet 4.6",
    "gpt-5.5-pro": "💎 GPT-5.5 Pro",
    "gpt-5.5": "🚀 GPT-5.5",
    "gpt-5.4": "⚡ GPT-5.4",
    "gpt-5.2-codex": "💻 GPT-5.2 Codex",
    "gpt-5.2-pro": "🌟 GPT-5.2 Pro",
    "gpt-5.2": "🤖 GPT-5.2",
}

# Combined for lookup
AVAILABLE_MODELS = {**AGENTROUTER_MODELS, **ATXP_MODELS, **FIREWORKS_MODELS}

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_PROVIDER = PROVIDER_AGENTROUTER
DEFAULT_SYSTEM_PROMPT = (
    "You are an advanced AI assistant similar to Claude Code CLI. "
    "You help with coding, analysis, writing, and complex problem solving. "
    "Be concise, accurate, and helpful. Format code with proper markdown."
)

# Conversation states
(
    WAITING_API_KEY,
    WAITING_ATXP_API_KEY,
    WAITING_FIREWORKS_API_KEY,
    WAITING_SYSTEM_PROMPT,
    WAITING_CUSTOM_MODEL,
    WAITING_DB_RESTORE,
) = range(6)


# ── Database ──────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id     INTEGER PRIMARY KEY,
            username    TEXT,
            api_key     TEXT,
            atxp_api_key TEXT,
            fireworks_api_key TEXT,
            provider    TEXT DEFAULT 'agentrouter',
            model       TEXT DEFAULT 'claude-haiku-4-5-20251001',
            system_prompt TEXT,
            created_at  TEXT DEFAULT (datetime('now')),
            updated_at  TEXT DEFAULT (datetime('now'))
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER,
            role        TEXT,
            content     TEXT,
            model       TEXT,
            created_at  TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS custom_models (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER,
            model_id    TEXT,
            model_name  TEXT,
            created_at  TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    """)

    conn.commit()
    conn.close()


def get_user(user_id: int) -> Optional[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None


def upsert_user(user_id: int, username: str = None, **kwargs):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    user = get_user(user_id)
    if not user:
        c.execute(
            "INSERT INTO users (user_id, username) VALUES (?, ?)",
            (user_id, username),
        )
    if kwargs:
        sets = ", ".join(f"{k}=?" for k in kwargs)
        vals = list(kwargs.values()) + [datetime.now().isoformat(), user_id]
        c.execute(f"UPDATE users SET {sets}, updated_at=? WHERE user_id=?", vals)
    conn.commit()
    conn.close()


def get_conversation(user_id: int, limit: int = 20) -> list:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute(
        "SELECT role, content FROM conversations WHERE user_id=? ORDER BY id DESC LIMIT ?",
        (user_id, limit),
    )
    rows = c.fetchall()
    conn.close()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def add_message(user_id: int, role: str, content: str, model: str = None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO conversations (user_id, role, content, model) VALUES (?,?,?,?)",
        (user_id, role, content, model),
    )
    conn.commit()
    conn.close()


def clear_conversation(user_id: int):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM conversations WHERE user_id=?", (user_id,))
    conn.commit()
    conn.close()


def get_custom_models(user_id: int) -> list:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM custom_models WHERE user_id=?", (user_id,))
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def add_custom_model(user_id: int, model_id: str, model_name: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO custom_models (user_id, model_id, model_name) VALUES (?,?,?)",
        (user_id, model_id, model_name),
    )
    conn.commit()
    conn.close()


# ── Backup / Restore ──────────────────────────────────────────────────────────
def export_db_base64() -> str:
    """Export entire SQLite DB as base64 string."""
    with open(DB_PATH, "rb") as f:
        return base64.b64encode(f.read()).decode()


def import_db_base64(b64: str):
    """Replace DB with base64 encoded content."""
    data = base64.b64decode(b64.encode())
    with open(DB_PATH, "wb") as f:
        f.write(data)


# ── AI API ────────────────────────────────────────────────────────────────────
async def call_ai(
    api_key: str,
    model: str,
    messages: list,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_tokens: int = 2048,
    provider: str = PROVIDER_AGENTROUTER,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    if provider == PROVIDER_ATXP:
        base_url = ATXP_BASE_URL
    elif provider == PROVIDER_FIREWORKS:
        base_url = FIREWORKS_BASE_URL
    else:
        base_url = f"{AGENTROUTER_BASE_URL}/v1"

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system_prompt}] + messages,
    }

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
        )

    if resp.status_code != 200:
        raise Exception(f"API Error {resp.status_code}: {resp.text[:500]}")

    raw = resp.text.strip()
    if not raw:
        raise Exception("Empty response from API.")

    try:
        data = resp.json()
    except Exception:
        raise Exception(f"Invalid JSON from API: {raw[:300]}")

    if "choices" in data and data["choices"]:
        return data["choices"][0]["message"]["content"]

    if "error" in data:
        raise Exception(f"API Error: {data['error']}")

    raise Exception(f"Unexpected response: {str(data)[:300]}")


# ── Keyboards ─────────────────────────────────────────────────────────────────
def main_menu_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🤖 AI Chat", callback_data="menu_chat"),
            InlineKeyboardButton("⚙️ Settings", callback_data="menu_settings"),
        ],
        [
            InlineKeyboardButton("🔄 Models", callback_data="menu_models"),
            InlineKeyboardButton("📜 History", callback_data="menu_history"),
        ],
        [
            InlineKeyboardButton("💾 Backup DB", callback_data="menu_backup"),
            InlineKeyboardButton("📤 Restore DB", callback_data="menu_restore"),
        ],
        [
            InlineKeyboardButton("🗑️ Clear Chat", callback_data="menu_clear"),
            InlineKeyboardButton("ℹ️ Help", callback_data="menu_help"),
        ],
    ])


def models_kb(user_id: int) -> InlineKeyboardMarkup:
    user = get_user(user_id)
    current_model = (user or {}).get("model", DEFAULT_MODEL)
    current_provider = (user or {}).get("provider", DEFAULT_PROVIDER)
    custom_models = get_custom_models(user_id)

    rows = []

    # AgentRouter models
    rows.append([InlineKeyboardButton("── 🌐 AgentRouter ──", callback_data="noop")])
    for model_id, label in AGENTROUTER_MODELS.items():
        check = "✅ " if (model_id == current_model and current_provider == PROVIDER_AGENTROUTER) else ""
        rows.append([InlineKeyboardButton(f"{check}{label}", callback_data=f"model_ar_{model_id}")])

    # ATXP models
    rows.append([InlineKeyboardButton("── 🔷 ATXP ──", callback_data="noop")])
    for model_id, label in ATXP_MODELS.items():
        check = "✅ " if (model_id == current_model and current_provider == PROVIDER_ATXP) else ""
        rows.append([InlineKeyboardButton(f"{check}{label}", callback_data=f"model_atxp_{model_id}")])

    # Fireworks models (short keys to stay under 64-byte Telegram limit)
    rows.append([InlineKeyboardButton("── 🔥 Fireworks ──", callback_data="noop")])
    for model_id, label in FIREWORKS_MODELS.items():
        check = "✅ " if (model_id == current_model and current_provider == PROVIDER_FIREWORKS) else ""
        short_key = FIREWORKS_MODEL_RMAP.get(model_id, model_id[:20])
        rows.append([InlineKeyboardButton(f"{check}{label}", callback_data=f"model_fw_{short_key}")])

    # Custom models
    if custom_models:
        rows.append([InlineKeyboardButton("── 🔧 Custom ──", callback_data="noop")])
        for cm in custom_models:
            check = "✅ " if cm["model_id"] == current_model else ""
            rows.append([InlineKeyboardButton(
                f"{check}🔧 {cm['model_name']}",
                callback_data=f"model_ar_{cm['model_id']}",
            )])

    rows.append([
        InlineKeyboardButton("➕ Add Custom Model", callback_data="add_custom_model"),
        InlineKeyboardButton("🔙 Back", callback_data="menu_main"),
    ])
    return InlineKeyboardMarkup(rows)


def settings_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔑 AgentRouter API Key", callback_data="set_api_key")],
        [InlineKeyboardButton("🔷 ATXP API Key", callback_data="set_atxp_api_key")],
        [InlineKeyboardButton("🔥 Fireworks API Key", callback_data="set_fireworks_api_key")],
        [InlineKeyboardButton("📝 Set System Prompt", callback_data="set_system_prompt")],
        [InlineKeyboardButton("🔙 Back", callback_data="menu_main")],
    ])


def back_kb(dest="menu_main") -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back", callback_data=dest)]])


# ── Handlers ──────────────────────────────────────────────────────────────────
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    upsert_user(user.id, user.username)
    text = (
        f"👋 *Welcome, {user.first_name}!*\n\n"
        "I'm an advanced AI bot powered by *AgentRouter.org*.\n"
        "I support multiple AI models and work like Claude Code CLI.\n\n"
        "👇 Choose an option to get started:"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=main_menu_kb())


async def button_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    user_id = query.from_user.id

    # ── Main Menu ──────────────────────────────────────────────────────────
    if data == "menu_main":
        await query.edit_message_text(
            "🏠 *Main Menu* — Choose an option:",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=main_menu_kb(),
        )

    elif data == "menu_chat":
        db_user = get_user(user_id)
        if not db_user or not db_user.get("api_key"):
            await query.edit_message_text(
                "⚠️ *API Key Required*\n\nPlease set your AgentRouter API key first.\n"
                "Get it from: https://agentrouter.org/console/token",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=settings_kb(),
            )
        else:
            model = db_user.get("model", DEFAULT_MODEL)
            model_name = AVAILABLE_MODELS.get(model, model)
            await query.edit_message_text(
                f"💬 *AI Chat Mode Active*\n\n"
                f"🤖 Model: `{model_name}`\n\n"
                f"Just send me any message and I'll respond!\n"
                f"Use /menu to return to the main menu.",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=back_kb(),
            )
            ctx.user_data["chat_mode"] = True

    elif data == "menu_settings":
        db_user = get_user(user_id) or {}
        api_set = "✅ Set" if db_user.get("api_key") else "❌ Not Set"
        prompt_set = "✅ Custom" if db_user.get("system_prompt") else "📋 Default"
        await query.edit_message_text(
            f"⚙️ *Settings*\n\n"
            f"🔑 API Key: {api_set}\n"
            f"📝 System Prompt: {prompt_set}\n"
            f"🤖 Model: `{db_user.get('model', DEFAULT_MODEL)}`",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=settings_kb(),
        )

    elif data == "menu_models":
        await query.edit_message_text(
            "🔄 *Select AI Model*\n\n✅ = Currently selected",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=models_kb(user_id),
        )

    elif data == "menu_history":
        msgs = get_conversation(user_id, limit=10)
        if not msgs:
            text = "📜 *No conversation history yet.*"
        else:
            lines = []
            for m in msgs[-6:]:
                role_icon = "👤" if m["role"] == "user" else "🤖"
                snippet = m["content"][:80].replace("\n", " ")
                lines.append(f"{role_icon} *{m['role'].capitalize()}:* {snippet}...")
            text = "📜 *Recent Conversation:*\n\n" + "\n\n".join(lines)
        await query.edit_message_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=back_kb())

    elif data == "menu_clear":
        clear_conversation(user_id)
        await query.edit_message_text(
            "🗑️ *Conversation history cleared!*\nYou can start a fresh conversation.",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=main_menu_kb(),
        )

    elif data == "menu_help":
        help_text = (
            "ℹ️ *Help & Commands*\n\n"
            "*/start* — Main menu\n"
            "*/menu* — Open menu anytime\n"
            "*/chat* — Start chatting\n"
            "*/clear* — Clear history\n"
            "*/backup* — Backup database\n"
            "*/restore* — Restore database\n\n"
            "🔑 *API Key*: Get from agentrouter.org/console/token\n\n"
            "🤖 *Models Available:*\n"
            + "\n".join(f"• {v}" for v in AVAILABLE_MODELS.values())
            + "\n\n💡 You can also add custom models in the Models menu!"
        )
        await query.edit_message_text(help_text, parse_mode=ParseMode.MARKDOWN, reply_markup=back_kb())

    # ── Backup ─────────────────────────────────────────────────────────────
    elif data == "menu_backup":
        try:
            b64 = export_db_base64()
            # Split into chunks if large (Telegram msg limit ~4096)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backup_{timestamp}.txt"

            # Send as file
            bio = io.BytesIO(b64.encode())
            bio.name = filename
            await query.message.reply_document(
                document=bio,
                filename=filename,
                caption=(
                    "💾 *Database Backup*\n\n"
                    "To restore: Send this file back or use /restore and paste the content."
                ),
                parse_mode=ParseMode.MARKDOWN,
            )
            await query.edit_message_text(
                "✅ *Backup created!*\nThe backup file has been sent above.",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=main_menu_kb(),
            )
        except Exception as e:
            await query.edit_message_text(
                f"❌ Backup failed: {e}",
                reply_markup=main_menu_kb(),
            )

    elif data == "menu_restore":
        await query.edit_message_text(
            "📤 *Restore Database*\n\n"
            "Send the backup file (the .txt file from backup) or paste the base64 content directly.\n\n"
            "⚠️ *Warning:* This will overwrite current data!\n\n"
            "Send /cancel to abort.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return WAITING_DB_RESTORE

    # ── Model Selection ────────────────────────────────────────────────────
    elif data.startswith("model_"):
        model_id = data[6:]
        upsert_user(user_id, model=model_id)
        model_name = AVAILABLE_MODELS.get(model_id, model_id)
        await query.edit_message_text(
            f"✅ *Model changed to:*\n`{model_name}`",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=models_kb(user_id),
        )

    elif data == "add_custom_model":
        await query.edit_message_text(
            "➕ *Add Custom Model*\n\n"
            "Send the model ID (e.g. `gpt-4o` or `mistral-large`)\n"
            "Format: `model_id|Model Display Name`\n\n"
            "Example: `gpt-4o|GPT-4o`\n\n"
            "Send /cancel to abort.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return WAITING_CUSTOM_MODEL

    # ── Settings ───────────────────────────────────────────────────────────
    elif data == "set_api_key":
        await query.edit_message_text(
            "🔑 *Set API Key*\n\n"
            "Send your AgentRouter API key.\n"
            "Get it from: https://agentrouter.org/console/token\n\n"
            "Your key starts with `sk-...`\n\n"
            "Send /cancel to abort.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return WAITING_API_KEY

    elif data == "set_system_prompt":
        await query.edit_message_text(
            "📝 *Set System Prompt*\n\n"
            "Send your custom system prompt, or send `default` to reset.\n\n"
            "Current default:\n"
            f"```\n{DEFAULT_SYSTEM_PROMPT[:200]}...\n```\n\n"
            "Send /cancel to abort.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return WAITING_SYSTEM_PROMPT


# ── Conversation Handlers ─────────────────────────────────────────────────────
async def handle_api_key(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    key = update.message.text.strip()
    user_id = update.effective_user.id

    if not key.startswith("sk-"):
        await update.message.reply_text(
            "❌ Invalid key format. API keys start with `sk-`\nTry again or /cancel",
            parse_mode=ParseMode.MARKDOWN,
        )
        return WAITING_API_KEY

    upsert_user(user_id, api_key=key)
    await update.message.reply_text(
        "✅ *API Key saved successfully!*\nYou can now start chatting.",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=main_menu_kb(),
    )
    # Delete the message containing the key for security
    try:
        await update.message.delete()
    except Exception:
        pass
    return ConversationHandler.END


async def handle_atxp_api_key(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    key = update.message.text.strip()
    user_id = update.effective_user.id

    upsert_user(user_id, atxp_api_key=key)
    await update.message.reply_text(
        "✅ *ATXP API Key saved!*\n\nNow select an ATXP model from 🔄 Models.",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=main_menu_kb(),
    )
    return ConversationHandler.END


async def handle_fireworks_api_key(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    key = update.message.text.strip()
    user_id = update.effective_user.id
    upsert_user(user_id, fireworks_api_key=key)
    await update.message.reply_text(
        "✅ *Fireworks API Key saved!*\n\nNow select a 🔥 Fireworks model from 🔄 Models.",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=main_menu_kb(),
    )
    return ConversationHandler.END


async def handle_system_prompt(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    user_id = update.effective_user.id

    if text.lower() == "default":
        upsert_user(user_id, system_prompt=None)
        await update.message.reply_text(
            "✅ *System prompt reset to default!*",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=main_menu_kb(),
        )
    else:
        upsert_user(user_id, system_prompt=text)
        await update.message.reply_text(
            f"✅ *System prompt saved!*\n\n```\n{text[:200]}\n```",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=main_menu_kb(),
        )
    return ConversationHandler.END


async def handle_custom_model(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    user_id = update.effective_user.id

    if "|" not in text:
        await update.message.reply_text(
            "❌ Wrong format. Use: `model_id|Display Name`\nExample: `gpt-4o|GPT-4o`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return WAITING_CUSTOM_MODEL

    model_id, model_name = text.split("|", 1)
    model_id = model_id.strip()
    model_name = model_name.strip()

    add_custom_model(user_id, model_id, model_name)
    await update.message.reply_text(
        f"✅ *Custom model added:*\n`{model_name}` (`{model_id}`)",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=models_kb(user_id),
    )
    return ConversationHandler.END


async def handle_db_restore(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    msg = update.message

    try:
        if msg.document:
            # File uploaded
            file = await ctx.bot.get_file(msg.document.file_id)
            bio = io.BytesIO()
            await file.download_to_memory(bio)
            b64 = bio.getvalue().decode().strip()
        else:
            b64 = msg.text.strip()

        import_db_base64(b64)
        await msg.reply_text(
            "✅ *Database restored successfully!*\nAll your data has been loaded.",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=main_menu_kb(),
        )
    except Exception as e:
        await msg.reply_text(
            f"❌ *Restore failed:*\n`{e}`\n\nMake sure you sent a valid backup file.",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=main_menu_kb(),
        )
    return ConversationHandler.END


async def cancel(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "❌ *Cancelled.*",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=main_menu_kb(),
    )
    return ConversationHandler.END


# ── AI Message Handler ────────────────────────────────────────────────────────
async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text

    db_user = get_user(user_id)
    if not db_user or not db_user.get("api_key"):
        await update.message.reply_text(
            "⚠️ Please set your API key first!\nUse /start → Settings → Set API Key",
            reply_markup=main_menu_kb(),
        )
        return

    # Show typing action
    await ctx.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    # Add user message to history
    add_message(user_id, "user", text, db_user.get("model", DEFAULT_MODEL))

    # Build conversation
    history = get_conversation(user_id, limit=20)
    model = db_user.get("model", DEFAULT_MODEL)
    system_prompt = db_user.get("system_prompt") or DEFAULT_SYSTEM_PROMPT

    # Send processing message
    thinking_msg = await update.message.reply_text("🤔 *Thinking...*", parse_mode=ParseMode.MARKDOWN)

    try:
        response = await call_ai(
            api_key=db_user["api_key"],
            model=model,
            messages=history,
            system_prompt=system_prompt,
        )

        # Save assistant response
        add_message(user_id, "assistant", response, model)

        model_label = AVAILABLE_MODELS.get(model, model)
        footer = f"\n\n─────────────\n🤖 `{model_label}`"

        # Handle long responses
        full_text = response + footer
        if len(full_text) > 4096:
            # Send in chunks
            await thinking_msg.delete()
            chunks = [full_text[i:i+4000] for i in range(0, len(full_text), 4000)]
            for i, chunk in enumerate(chunks):
                if i == 0:
                    await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)
                else:
                    await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)
        else:
            await thinking_msg.edit_text(full_text, parse_mode=ParseMode.MARKDOWN)

    except Exception as e:
        logger.error(f"API Error for user {user_id}: {e}")
        await thinking_msg.edit_text(
            f"❌ *Error:* `{str(e)[:200]}`\n\nCheck your API key and try again.",
            parse_mode=ParseMode.MARKDOWN,
        )


# ── Command Shortcuts ─────────────────────────────────────────────────────────
async def cmd_menu(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🏠 *Main Menu*",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=main_menu_kb(),
    )


async def cmd_clear(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    clear_conversation(update.effective_user.id)
    await update.message.reply_text(
        "🗑️ *History cleared!*",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=main_menu_kb(),
    )


async def cmd_backup(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        b64 = export_db_base64()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bio = io.BytesIO(b64.encode())
        bio.name = f"backup_{timestamp}.txt"
        await update.message.reply_document(
            document=bio,
            filename=f"backup_{timestamp}.txt",
            caption="💾 *Database Backup*\nKeep this file safe to restore later!",
            parse_mode=ParseMode.MARKDOWN,
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Backup failed: {e}")


async def cmd_restore(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📤 *Restore Database*\n\nSend the backup .txt file or paste the base64 content.\n\n"
        "⚠️ This will overwrite current data!\n\nSend /cancel to abort.",
        parse_mode=ParseMode.MARKDOWN,
    )
    return WAITING_DB_RESTORE


# ── Dummy HTTP Server (Render-এর জন্য) ───────────────────────────────────────
class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Bot is running!")

    def log_message(self, format, *args):
        pass  # HTTP log বন্ধ রাখো


def run_health_server():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    logger.info(f"Health server running on port {port}")
    server.serve_forever()


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required!")

    init_db()
    logger.info("Database initialized.")

    app = Application.builder().token(token).build()

    # Conversation handler for setup flows
    conv_handler = ConversationHandler(
        entry_points=[
            CallbackQueryHandler(button_handler, pattern="^(set_api_key|set_atxp_api_key|set_fireworks_api_key|set_system_prompt|add_custom_model|menu_restore)$"),
            CommandHandler("restore", cmd_restore),
        ],
        per_message=False,
        states={
            WAITING_API_KEY: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_api_key)],
            WAITING_ATXP_API_KEY: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_atxp_api_key)],
            WAITING_FIREWORKS_API_KEY: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_fireworks_api_key)],
            WAITING_SYSTEM_PROMPT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_system_prompt)],
            WAITING_CUSTOM_MODEL: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_custom_model)],
            WAITING_DB_RESTORE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_db_restore),
                MessageHandler(filters.Document.ALL, handle_db_restore),
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    # Register handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("menu", cmd_menu))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("backup", cmd_backup))
    app.add_handler(conv_handler)
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot started! Polling...")
    async with app:
        await app.initialize()
        await app.start()
        await app.updater.start_polling(drop_pending_updates=True)
        # Keep running until interrupted
        await asyncio.Event().wait()


if __name__ == "__main__":
    # Dummy HTTP server আলাদা thread-এ চালাও
    t = threading.Thread(target=run_health_server, daemon=True)
    t.start()
    asyncio.run(main())
