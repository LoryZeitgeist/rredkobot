import json
import logging
import os
import platform
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from telegram import (
    BotCommand,
    BotCommandScopeAllPrivateChats,
    BotCommandScopeChat,
    BotCommandScopeDefault,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.constants import ChatType
from telegram.ext import (
    AIORateLimiter,
    Application,
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    MessageHandler,
    filters,
)


load_dotenv()

logger = logging.getLogger("bot")

BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
ADMIN_IDS = {
    int(admin_id.strip())
    for admin_id in os.environ.get("ADMIN_IDS", "").split(",")
    if admin_id.strip().isdigit()
}
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
PRESETS_DIR = Path(os.environ.get("PRESETS_DIR", "/app/presets")).resolve()
STATE_FILE = Path(os.environ.get("STATE_FILE", "/app/data/chat_presets.json")).resolve()
ENV_FILE_PATH = Path(".env").resolve()

CMD_START = "start"
CMD_HELP = "spravka"
CMD_LIST_PRESETS = "spisok_presetov"
CMD_RELOAD_PRESETS = "obnovit_presets"
CMD_APPLY_PRESET = "naznachit_preset"
CMD_SHOW_STATE = "tekuschee_sostoyanie"
CMD_DEBUG_STATUS = "diagnostika_okruzheniya"
CMD_DEBUG_CHAT = "diagnostika_chata"
CMD_LIST_TRIGGERS = "spisok_triggers"
CMD_ADMIN_IMAGES = "upravlenie_kartinkami"

COMMAND_ALIASES = {
    CMD_HELP: ["help"],
    CMD_LIST_PRESETS: ["list_presets"],
    CMD_RELOAD_PRESETS: ["reload_presets"],
    CMD_APPLY_PRESET: ["apply_preset"],
    CMD_SHOW_STATE: ["show_state"],
    CMD_DEBUG_STATUS: ["debug_status"],
    CMD_DEBUG_CHAT: ["debug_chat"],
    CMD_LIST_TRIGGERS: ["list_triggers"],
    CMD_ADMIN_IMAGES: ["admin_images"],
}

def command_with_aliases(command: str) -> List[str]:
    return [command, *COMMAND_ALIASES.get(command, [])]

USER_COMMANDS = [
    BotCommand(CMD_START, "Получить информацию о боте"),
    BotCommand(CMD_HELP, "Список доступных команд"),
]

ADMIN_ONLY_COMMANDS = [
    BotCommand(CMD_LIST_PRESETS, "Показать все пресеты"),
    BotCommand(CMD_RELOAD_PRESETS, "Перезагрузить пресеты"),
    BotCommand(CMD_APPLY_PRESET, "Назначить пресет чату"),
    BotCommand(CMD_SHOW_STATE, "Показать назначения пресетов"),
    BotCommand(CMD_DEBUG_STATUS, "Диагностика окружения"),
    BotCommand(CMD_DEBUG_CHAT, "Диагностика чата"),
    BotCommand(CMD_LIST_TRIGGERS, "Показать триггеры пресета"),
    BotCommand(CMD_ADMIN_IMAGES, "Управление картинками пресетов"),
]

ADMIN_COMMANDS = USER_COMMANDS + ADMIN_ONLY_COMMANDS


def setup_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def ensure_directories() -> None:
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

def calculate_min_matches(keywords: List[str], configured_min: Optional[int]) -> int:
    if configured_min is not None:
        if isinstance(configured_min, int) and configured_min > 0:
            return min(configured_min, len(keywords)) if keywords else configured_min
    if len(keywords) >= 2:
        return 2
    return len(keywords)


def log_startup_diagnostics(presets: Dict[str, Dict[str, Any]], state: Dict[str, str]) -> None:
    logger.info(
        "Startup diagnostics: platform=%s, cwd=%s, env_file=%s (exists=%s), presets_dir=%s (exists=%s), "
        "state_file=%s (exists=%s), bot_token_present=%s, admin_ids=%s",
        platform.platform(),
        Path.cwd(),
        ENV_FILE_PATH,
        ENV_FILE_PATH.exists(),
        PRESETS_DIR,
        PRESETS_DIR.exists(),
        STATE_FILE,
        STATE_FILE.exists(),
        bool(BOT_TOKEN),
        sorted(ADMIN_IDS),
    )
    logger.info("Loaded %d presets: %s", len(presets), list(presets.keys()))
    logger.info("Loaded state assignments: %s", state)
    if not ADMIN_IDS:
        logger.warning("ADMIN_IDS environment variable is empty. Админские команды будут недоступны.")


def load_presets() -> Dict[str, Dict[str, Any]]:
    presets: Dict[str, Dict[str, Any]] = {}

    if not PRESETS_DIR.exists():
        logger.warning("Preset directory %s not found.", PRESETS_DIR)
        return presets

    for preset_path in PRESETS_DIR.glob("*.json"):
        try:
            preset_data = json.loads(preset_path.read_text(encoding="utf-8"))
            name = preset_data.get("name") or preset_path.stem
            presets[name] = preset_data
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse preset %s: %s", preset_path, exc)

    return presets


def load_state() -> Dict[str, str]:
    if not STATE_FILE.exists():
        return {}

    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        if isinstance(state, dict):
            return {str(k): str(v) for k, v in state.items()}
    except json.JSONDecodeError as exc:
        logger.error("Failed to load state from %s: %s", STATE_FILE, exc)

    return {}


def save_state(state: Dict[str, str]) -> None:
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_image_path(image_ref: str) -> Optional[Path]:
    candidate = (PRESETS_DIR / image_ref).resolve()
    try:
        candidate.relative_to(PRESETS_DIR)
    except ValueError:
        logger.error("Image %s is outside of presets directory %s.", image_ref, PRESETS_DIR)
        return None

    if not candidate.exists():
        logger.error("Image file %s not found (resolved path %s).", image_ref, candidate)
        return None

    if not candidate.is_file():
        logger.error("Image path %s is not a file.", candidate)
        return None

    return candidate


def generate_state_keys(chat_id: int) -> List[str]:
    str_id = str(chat_id)
    keys = [str_id]

    if str_id.startswith("-100") and len(str_id) > 4:
        keys.append(str_id[4:])

    if str_id.startswith("-") and len(str_id) > 1:
        keys.append(str_id[1:])

    abs_id = str(abs(chat_id))
    if abs_id not in keys:
        keys.append(abs_id)

    return keys


def get_chat_preset_name(chat_id: int, state: Dict[str, str]) -> Optional[str]:
    for key in generate_state_keys(chat_id):
        if key in state:
            return state[key]
    return None


def find_matching_trigger(
    message_text: Optional[str],
    preset: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not message_text:
        return None

    normalized_message = message_text.lower()
    for index, trigger in enumerate(preset.get("triggers", [])):
        phrase = str(trigger.get("phrase", "")).strip().lower()
        explicit_keywords = [
            str(keyword).strip().lower()
            for keyword in trigger.get("keywords", [])
            if str(keyword).strip()
        ]
        derived_keywords = [
            token.strip()
            for token in phrase.split()
            if token.strip()
        ] if phrase else []

        keywords: List[str] = []
        seen = set()
        for keyword in (explicit_keywords or derived_keywords):
            if keyword and keyword not in seen:
                seen.add(keyword)
                keywords.append(keyword)

        if len(keywords) < 2:
            logger.debug("Trigger %s skipped: требуется минимум два ключевых слова.", trigger)
            continue

        matches = sum(1 for keyword in keywords if keyword in normalized_message)
        min_matches = max(2, calculate_min_matches(keywords, trigger.get("min_matches")))
        if matches >= min_matches:
            return {
                "trigger": trigger,
                "index": index,
                "meta": {
                    "type": "keywords",
                    "keywords": keywords,
                    "matches": matches,
                    "min_matches": min_matches,
                },
            }
    return None


def collect_trigger_images(trigger: Dict[str, Any]) -> List[Path]:
    image_paths: List[Path] = []
    for image_ref in trigger.get("images", []):
        if not isinstance(image_ref, str):
            logger.warning("Skipping non-string image reference %s in trigger %s.", image_ref, trigger)
            continue
        path = resolve_image_path(image_ref)
        if path:
            image_paths.append(path)
    return image_paths


def get_next_trigger_image(
    chat_id: int,
    preset_name: str,
    trigger_index: int,
    trigger: Dict[str, Any],
    bot_data: Dict[str, Any],
) -> Optional[Path]:
    image_paths = collect_trigger_images(trigger)
    if not image_paths:
        logger.error("Trigger %s has no valid image files in %s.", trigger, PRESETS_DIR)
        return None

    key = f"{chat_id}:{preset_name}:{trigger_index}"
    image_cycles = bot_data.setdefault("image_cycles", {})
    entry = image_cycles.get(key)

    current_files = [str(path) for path in image_paths]

    if (
        not entry
        or sorted(entry.get("files", [])) != sorted(current_files)
        or not entry.get("remaining")
    ):
        shuffled = current_files[:]
        random.shuffle(shuffled)
        entry = {"files": current_files, "remaining": shuffled}
        image_cycles[key] = entry

    if not entry["remaining"]:
        shuffled = current_files[:]
        random.shuffle(shuffled)
        entry["remaining"] = shuffled

    next_file = entry["remaining"].pop()
    path = Path(next_file)

    if not path.exists():
        logger.warning("File %s missing on disk; resetting image cycle.", path)
        image_cycles.pop(key, None)
        return get_next_trigger_image(chat_id, preset_name, trigger_index, trigger, bot_data)

    return path


def is_admin(update: Update) -> bool:
    user = update.effective_user
    return bool(user and user.id in ADMIN_IDS)


def is_private_chat(update: Update) -> bool:
    chat = update.effective_chat
    return bool(chat and chat.type == ChatType.PRIVATE)


def get_available_commands(is_admin_user: bool) -> List[BotCommand]:
    return ADMIN_COMMANDS if is_admin_user else USER_COMMANDS


def ensure_presets_directory() -> None:
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)


def build_admin_images_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("🖼 Загрузить пресет", callback_data="admin_presets:upload")],
            [InlineKeyboardButton("📋 Список пресетов", callback_data="admin_presets:list")],
            [InlineKeyboardButton("🗑 Удалить пресет", callback_data="admin_presets:delete")],
            [InlineKeyboardButton("🔙 Назад / Выход", callback_data="admin_presets:exit")],
        ]
    )


def build_files_keyboard(action: str, files: List[Path], *, use_index_labels: bool = False) -> InlineKeyboardMarkup:
    buttons = [
        [
            InlineKeyboardButton(
                f"🖼 {index + 1}" if use_index_labels else f"📂 {file_path.name}",
                callback_data=f"{action}:{file_path.name}",
            )
        ]
        for index, file_path in enumerate(files)
    ]
    buttons.append([InlineKeyboardButton("🔙 Назад", callback_data="admin_presets:menu")])
    return InlineKeyboardMarkup(buttons)


def build_back_to_menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Назад", callback_data="admin_presets:menu")]])


def list_preset_files() -> List[Path]:
    ensure_presets_directory()
    allowed = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(
        file_path
        for file_path in PRESETS_DIR.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in allowed
    )


async def ensure_admin_panel_access(update: Update) -> bool:
    if not is_admin(update):
        if update.callback_query:
            await update.callback_query.answer("🚫 У вас нет доступа к админ-панели.", show_alert=True)
        elif update.effective_message:
            await update.effective_message.reply_text("🚫 У вас нет доступа к админ-панели.")
        return False

    if not is_private_chat(update):
        if update.callback_query:
            await update.callback_query.answer("Админ-панель доступна только в личном чате.", show_alert=True)
        elif update.effective_message:
            await update.effective_message.reply_text("Админ-панель доступна только в личном чате.")
        return False

    return True


async def show_admin_presets_menu(update: Update, context: ContextTypes.DEFAULT_TYPE, *, edit: bool = False) -> None:
    keyboard = build_admin_images_keyboard()
    text = "Главное меню админ-панели пресетов:\nВыберите действие:"
    context.user_data.pop("awaiting_preset_upload", None)

    if edit and update.callback_query and update.callback_query.message:
        await update.callback_query.message.edit_text(text, reply_markup=keyboard)
        await update.callback_query.answer()
    else:
        await update.effective_message.reply_text(text, reply_markup=keyboard)


async def cmd_admin_images(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await ensure_admin_panel_access(update):
        return

    logger.info("Admin %s opened image admin panel.", update.effective_user.id if update.effective_user else "unknown")
    await show_admin_presets_menu(update, context)


async def handle_admin_presets_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await ensure_admin_panel_access(update):
        return

    query = update.callback_query
    if not query:
        return

    data = query.data or ""
    logger.debug("Admin %s triggered callback %s.", update.effective_user.id if update.effective_user else "unknown", data)

    if data in {"admin_presets:menu", "admin_presets:exit"}:
        await show_admin_presets_menu(update, context, edit=True)
        return

    if data == "admin_presets:upload":
        context.user_data["awaiting_preset_upload"] = True
        await query.message.edit_text(
            "Отправьте изображение (фото или файл). После загрузки оно появится в папке presets/.",
            reply_markup=build_back_to_menu_keyboard(),
        )
        await query.answer()
        return

    if data == "admin_presets:list":
        logger.debug("Admin %s requested file list for viewing.", update.effective_user.id if update.effective_user else "unknown")
        files = list_preset_files()
        if not files:
            await query.message.edit_text(
                "В папке presets пока нет файлов.",
                reply_markup=build_back_to_menu_keyboard(),
            )
            await query.answer()
            return

        await query.message.edit_text(
            "Сохранённые пресеты:\nНажмите на кнопку, чтобы посмотреть изображение.",
            reply_markup=build_files_keyboard("admin_presets:view", files, use_index_labels=True),
        )
        await query.answer()
        return

    if data.startswith("admin_presets:view:"):
        filename = data.split(":", 2)[2]
        file_path = resolve_image_path(filename)
        if not file_path:
            await query.answer("Файл не найден.", show_alert=True)
            return

        with file_path.open("rb") as image_file:
            await query.message.reply_photo(photo=image_file)
        logger.info(
            "Admin %s requested preset image %s.",
            update.effective_user.id if update.effective_user else "unknown",
            file_path,
        )
        await query.answer()
        return

    if data == "admin_presets:delete":
        logger.debug("Admin %s requested file list for deletion.", update.effective_user.id if update.effective_user else "unknown")
        files = list_preset_files()
        if not files:
            await query.message.edit_text(
                "Нет файлов для удаления.",
                reply_markup=build_back_to_menu_keyboard(),
            )
            await query.answer()
            return

        await query.message.edit_text(
            "Выберите картинку для удаления:",
            reply_markup=build_files_keyboard("admin_presets:remove", files),
        )
        await query.answer()
        return

    if data.startswith("admin_presets:remove:"):
        filename = data.split(":", 2)[2]
        file_path = resolve_image_path(filename)
        if not file_path:
            await query.answer("Файл не найден.", show_alert=True)
            return

        try:
            file_path.unlink()
        except OSError as exc:
            await query.answer(f"Не удалось удалить файл: {exc}", show_alert=True)
            return

        await query.message.edit_text(
            f"🗑 Пресет {filename} удалён.",
            reply_markup=build_admin_images_keyboard(),
        )
        context.user_data.pop("awaiting_preset_upload", None)
        logger.info(
            "Admin %s deleted preset image %s.",
            update.effective_user.id if update.effective_user else "unknown",
            file_path,
        )
        await query.answer()
        return

    await query.answer()


async def handle_admin_image_upload(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.user_data.get("awaiting_preset_upload"):
        return

    if not await ensure_admin_panel_access(update):
        return

    message = update.effective_message
    if not message:
        return

    ensure_presets_directory()
    filename = f"preset_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
    destination = PRESETS_DIR / filename

    telegram_file = None
    if message.photo:
        telegram_file = await message.photo[-1].get_file()
    elif message.document and (
        (message.document.mime_type and message.document.mime_type.startswith("image/"))
        or (message.document.file_name and message.document.file_name.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".webp")))
    ):
        telegram_file = await message.document.get_file()

    if not telegram_file:
        await message.reply_text("⚠️ Пожалуйста, отправьте изображение (фото или файл).")
        return

    await telegram_file.download_to_drive(custom_path=str(destination))
    context.user_data.pop("awaiting_preset_upload", None)

    logger.info(
        "Admin %s uploaded new preset image %s.",
        update.effective_user.id if update.effective_user else "unknown",
        destination,
    )

    await message.reply_text(f"✅ Картинка сохранена в presets/ как {destination.name}.")
    await message.reply_text(
        "Возвращаю вас в меню админки:",
        reply_markup=build_admin_images_keyboard(),
    )


async def handle_admin_non_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.user_data.get("awaiting_preset_upload"):
        return

    if not await ensure_admin_panel_access(update):
        return

    message = update.effective_message
    if message:
        await message.reply_text("⚠️ Пожалуйста, отправьте изображение (фото или файл).")


async def configure_chat_commands(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    if not chat:
        return

    is_admin_user = is_admin(update)
    commands = get_available_commands(is_admin_user)
    try:
        await context.bot.set_my_commands(commands, scope=BotCommandScopeChat(chat.id))
        logger.debug(
            "Updated command list for chat %s (admin=%s): %s",
            chat.id,
            is_admin_user,
            [command.command for command in commands],
        )
    except Exception as exc:
        logger.error("Failed to set commands for chat %s: %s", chat.id, exc)


async def initialize_command_suggestions(application: Application) -> None:
    try:
        await application.bot.set_my_commands(USER_COMMANDS, scope=BotCommandScopeDefault())
        await application.bot.set_my_commands(USER_COMMANDS, scope=BotCommandScopeAllPrivateChats())
        logger.info("Default command suggestions registered.")
    except Exception as exc:
        logger.error("Failed to initialize default command suggestions: %s", exc)


async def handle_group_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    message = update.effective_message
    user = update.effective_user
    if not chat or not message:
        return

    if chat.type not in {ChatType.GROUP, ChatType.SUPERGROUP}:
        return

    logger.info(
        "Received message in chat %s by user %s (%s): %s",
        chat.id,
        user.full_name if user else "Unknown",
        user.id if user else "unknown",
        message.text,
    )

    state = context.bot_data.setdefault("state", {})
    preset_name = get_chat_preset_name(chat.id, state)
    if not preset_name:
        logger.warning(
            "Chat %s has no preset assigned. Checked keys %s. Current assignments: %s",
            chat.id,
            generate_state_keys(chat.id),
            state,
        )
        return

    presets = context.bot_data.setdefault("presets", {})
    preset = presets.get(preset_name)
    if not preset:
        logger.warning("Preset %s assigned to chat %s is missing. Available presets: %s.", preset_name, chat.id, list(presets.keys()))
        return

    match = find_matching_trigger(message.text, preset)
    if not match:
        logger.debug(
            "Chat %s message '%s' did not match triggers in preset %s.",
            chat.id,
            message.text,
            preset_name,
        )
        return

    trigger = match["trigger"]
    trigger_index = match.get("index", 0)
    meta = match["meta"]

    image_path = get_next_trigger_image(chat.id, preset_name, trigger_index, trigger, context.bot_data)
    if not image_path:
        logger.warning("Trigger %s in preset %s has no usable image files.", trigger, preset_name)
        return

    logger.info(
        "Sending image for chat %s using preset %s trigger '%s' (match=%s, file=%s).",
        chat.id,
        preset_name,
        trigger.get("phrase"),
        meta,
        image_path,
    )
    with image_path.open("rb") as image_file:
        await message.reply_photo(photo=image_file)


def admin_only(func):
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not is_admin(update):
            logger.info("Unauthorized access attempt by user %s", update.effective_user)
            return
        if not is_private_chat(update):
            await update.effective_message.reply_text("Пожалуйста, используйте эту команду в личных сообщениях со мной.")
            return
        logger.debug("Admin %s invoked %s.", update.effective_user.id, func.__name__)
        return await func(update, context)

    return wrapper


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    is_admin_user = is_admin(update)
    await configure_chat_commands(update, context)
    commands_text = "\n".join(
        f"/{command.command} — {command.description}" for command in get_available_commands(is_admin_user)
    )
    logger.info(
        "User %s (%s) invoked /start in chat %s (admin=%s).",
        update.effective_user.full_name if update.effective_user else "Unknown",
        update.effective_user.id if update.effective_user else "unknown",
        update.effective_chat.id if update.effective_chat else "unknown",
        is_admin_user,
    )
    await update.effective_message.reply_text(
        "Привет! Я реагирую на сообщения в группе по ключевым словам из назначенного пресета.\n\n"
        "Доступные команды:\n"
        f"{commands_text}"
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    is_admin_user = is_admin(update)
    commands = get_available_commands(is_admin_user)
    commands_text = "\n".join(f"/{cmd.command} — {cmd.description}" for cmd in commands)
    logger.info(
        "User %s (%s) requested /%s in chat %s (admin=%s).",
        update.effective_user.full_name if update.effective_user else "Unknown",
        update.effective_user.id if update.effective_user else "unknown",
        update.effective_chat.id if update.effective_chat else "unknown",
        CMD_HELP,
        is_admin_user,
    )
    await update.effective_message.reply_text(
        "Доступные команды:\n"
        f"{commands_text}"
    )


@admin_only
async def cmd_list_presets(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    presets = context.bot_data.setdefault("presets", {})
    logger.info(
        "Admin %s requested preset list. %d presets available.",
        update.effective_user.id,
        len(presets),
    )
    if not presets:
        await update.effective_message.reply_text("Пресеты не найдены. Добавьте JSON-файлы в папку presets.")
        return

    lines = []
    for preset_name, preset in presets.items():
        description = preset.get("description", "")
        lines.append(f"- {preset_name} {f'({description})' if description else ''}".rstrip())

    await update.effective_message.reply_text("\n".join(lines))


@admin_only
async def cmd_reload_presets(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    presets = load_presets()
    context.bot_data["presets"] = presets
    logger.info(
        "Presets reloaded by admin %s. Available presets: %s",
        update.effective_user.id,
        list(presets.keys()),
    )
    await update.effective_message.reply_text("Пресеты перезагружены.")


@admin_only
async def cmd_apply_preset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    args = context.args
    if len(args) != 2:
        await message.reply_text(f"Использование: /{CMD_APPLY_PRESET} <chat_id> <preset_name>")
        return

    chat_id_str, preset_name = args
    if not chat_id_str.lstrip("-").isdigit():
        await message.reply_text("chat_id должен быть числом.")
        return

    chat_id = int(chat_id_str)
    presets = context.bot_data.setdefault("presets", {})
    if preset_name not in presets:
        await message.reply_text(f"Пресет {preset_name} не найден. Используйте /{CMD_LIST_PRESETS}.")
        return

    state = context.bot_data.setdefault("state", {})
    state[str(chat_id)] = preset_name
    save_state(state)
    logger.info(
        "Admin %s applied preset %s to chat %s.",
        update.effective_user.id,
        preset_name,
        chat_id,
    )
    await message.reply_text(f"Пресет {preset_name} применён к чату {chat_id}.")


@admin_only
async def cmd_show_state(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = context.bot_data.setdefault("state", {})
    if not state:
        await update.effective_message.reply_text("Пока ни к одному чату не применён пресет.")
        return

    lines = [f"{chat_id}: {preset_name}" for chat_id, preset_name in state.items()]
    await update.effective_message.reply_text("\n".join(lines))


@admin_only
async def cmd_debug_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    presets = context.bot_data.setdefault("presets", {})
    state = context.bot_data.setdefault("state", {})
    diagnostics = [
        f"Токен загружен: {'да' if BOT_TOKEN else 'нет'}",
        f"Путь до .env: {ENV_FILE_PATH} (существует: {'да' if ENV_FILE_PATH.exists() else 'нет'})",
        f"Папка пресетов: {PRESETS_DIR} (существует: {'да' if PRESETS_DIR.exists() else 'нет'})",
        f"Файл состояния: {STATE_FILE} (существует: {'да' if STATE_FILE.exists() else 'нет'})",
        f"Количество пресетов: {len(presets)}",
        f"Назначения пресетов: {state}",
        f"Рабочая директория: {Path.cwd()}",
        f"Переменные окружения: LOG_LEVEL={LOG_LEVEL}, ADMIN_IDS={sorted(ADMIN_IDS)}",
        f"Платформа контейнера: {platform.platform()}",
    ]
    logger.info("Admin %s requested debug status.", update.effective_user.id)
    await update.effective_message.reply_text("\n".join(diagnostics))


@admin_only
async def cmd_debug_chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = context.bot_data.setdefault("state", {})
    args = context.args
    if not args:
        await update.effective_message.reply_text(f"Использование: /{CMD_DEBUG_CHAT} <chat_id>")
        return

    chat_id = args[0]
    assigned_preset = state.get(chat_id)
    logger.info(
        "Admin %s requested debug chat info for chat %s. Assigned preset: %s",
        update.effective_user.id,
        chat_id,
        assigned_preset,
    )
    await update.effective_message.reply_text(
        f"Чат: {chat_id}\nПресет: {assigned_preset or 'не назначен'}"
    )


@admin_only
async def cmd_list_triggers(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    presets = context.bot_data.setdefault("presets", {})
    args = context.args
    if not args:
        await update.effective_message.reply_text(f"Использование: /{CMD_LIST_TRIGGERS} <preset_name>")
        return

    preset_name = args[0]
    preset = presets.get(preset_name)
    if not preset:
        await update.effective_message.reply_text("Пресет не найден.")
        return

    lines: List[str] = []
    for trigger in preset.get("triggers", []):
        phrase = str(trigger.get("phrase", "")).strip()
        keywords = [str(keyword).strip() for keyword in trigger.get("keywords", []) if str(keyword).strip()]
        if keywords:
            required = max(2, calculate_min_matches(keywords, trigger.get("min_matches")))
            warning = ""
            if len(keywords) < required:
                warning = " (предупреждение: ключевых слов меньше, чем требуется для совпадения)"
            lines.append(f"keywords={keywords}, min_matches={required}{warning}")
        elif phrase:
            lines.append(f"phrase='{phrase}'")
        else:
            lines.append("Некорректный триггер (нет phrase/keywords)")

    logger.info(
        "Admin %s requested triggers for preset %s. %d triggers found.",
        update.effective_user.id,
        preset_name,
        len(lines),
    )
    await update.effective_message.reply_text("\n".join(lines) if lines else "Триггеры не найдены.")


def build_application() -> Application:
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN is not set. Please configure the .env file.")

    setup_logging()
    ensure_directories()

    presets = load_presets()
    state = load_state()
    log_startup_diagnostics(presets, state)

    application = (
        ApplicationBuilder()
        .token(BOT_TOKEN)
        .rate_limiter(AIORateLimiter())
        .post_init(initialize_command_suggestions)
        .build()
    )

    application.bot_data["presets"] = presets
    application.bot_data["state"] = state

    application.add_handler(CommandHandler(CMD_START, cmd_start))
    application.add_handler(CommandHandler(command_with_aliases(CMD_HELP), cmd_help))
    application.add_handler(CommandHandler(command_with_aliases(CMD_LIST_PRESETS), cmd_list_presets))
    application.add_handler(CommandHandler(command_with_aliases(CMD_RELOAD_PRESETS), cmd_reload_presets))
    application.add_handler(CommandHandler(command_with_aliases(CMD_APPLY_PRESET), cmd_apply_preset))
    application.add_handler(CommandHandler(command_with_aliases(CMD_SHOW_STATE), cmd_show_state))
    application.add_handler(CommandHandler(command_with_aliases(CMD_DEBUG_STATUS), cmd_debug_status))
    application.add_handler(CommandHandler(command_with_aliases(CMD_DEBUG_CHAT), cmd_debug_chat))
    application.add_handler(CommandHandler(command_with_aliases(CMD_LIST_TRIGGERS), cmd_list_triggers))
    application.add_handler(CommandHandler(command_with_aliases(CMD_ADMIN_IMAGES), cmd_admin_images))
    application.add_handler(CallbackQueryHandler(handle_admin_presets_callback, pattern=r"^admin_presets:"))
    application.add_handler(
        MessageHandler(
            filters.ChatType.PRIVATE & (filters.PHOTO | filters.Document.IMAGE),
            handle_admin_image_upload,
        )
    )
    application.add_handler(
        MessageHandler(
            filters.ChatType.PRIVATE & filters.TEXT & (~filters.COMMAND),
            handle_admin_non_image,
        )
    )

    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_group_message))

    return application


def main() -> None:
    application = build_application()
    logger.info("Bot starting.")
    application.run_polling()


if __name__ == "__main__":
    main()
