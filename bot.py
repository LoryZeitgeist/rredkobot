import json
import logging
import os
import platform
import random
import shutil
import tempfile
import zipfile
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
ALLOWED_ARCHIVE_EXTENSIONS = {".zip"}
PRESET_FILE_MAP: Dict[str, Path] = {}

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
    PRESET_FILE_MAP.clear()

    if not PRESETS_DIR.exists():
        logger.warning("Preset directory %s not found.", PRESETS_DIR)
        return presets

    for preset_path in PRESETS_DIR.glob("*.json"):
        try:
            preset_data = json.loads(preset_path.read_text(encoding="utf-8"))
            name = preset_data.get("name") or preset_path.stem
            presets[name] = preset_data
            PRESET_FILE_MAP[name] = preset_path
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

        if not keywords:
            logger.debug("Trigger %s skipped: нет доступных ключевых слов или фразы.", trigger)
            continue

        matches = sum(1 for keyword in keywords if keyword in normalized_message)
        min_matches = calculate_min_matches(keywords, trigger.get("min_matches"))
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


def get_preset_file_path(preset_name: str) -> Optional[Path]:
    preset_path = PRESET_FILE_MAP.get(preset_name)
    if preset_path and preset_path.exists():
        return preset_path
    logger.error("Preset file for %s not found. Known presets: %s", preset_name, list(PRESET_FILE_MAP.keys()))
    return None


def append_image_references_to_preset(preset_name: str, trigger_index: int, filenames: List[str]) -> None:
    preset_path = get_preset_file_path(preset_name)
    if not preset_path:
        raise FileNotFoundError(f"Не найден JSON-файл пресета {preset_name}.")

    try:
        data = json.loads(preset_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Не удалось прочитать {preset_path}: {exc}") from exc

    triggers = data.setdefault("triggers", [])
    if trigger_index >= len(triggers):
        raise IndexError("Выбранный триггер не существует.")

    trigger = triggers[trigger_index]
    images = trigger.setdefault("images", [])
    for filename in filenames:
        if filename not in images:
            images.append(filename)

    preset_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_keywords_from_phrase(phrase: str) -> List[str]:
    keywords: List[str] = []
    for raw_token in phrase.split():
        token = raw_token.strip().lower()
        if token and token not in keywords:
            keywords.append(token)
    return keywords


def update_trigger_phrase_in_preset(preset_name: str, trigger_index: int, new_phrase: str) -> None:
    preset_path = get_preset_file_path(preset_name)
    if not preset_path:
        raise FileNotFoundError(f"Не найден JSON-файл пресета {preset_name}.")

    try:
        data = json.loads(preset_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Не удалось прочитать {preset_path}: {exc}") from exc

    triggers = data.setdefault("triggers", [])
    if trigger_index < 0 or trigger_index >= len(triggers):
        raise IndexError("Выбранный триггер не существует.")

    triggers[trigger_index]["phrase"] = new_phrase
    new_keywords = extract_keywords_from_phrase(new_phrase)
    if new_keywords:
        triggers[trigger_index]["keywords"] = new_keywords
    preset_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def remove_image_from_trigger(preset_name: str, trigger_index: int, image_index: int) -> str:
    preset_path = get_preset_file_path(preset_name)
    if not preset_path:
        raise FileNotFoundError(f"Не найден JSON-файл пресета {preset_name}.")

    try:
        data = json.loads(preset_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Не удалось прочитать {preset_path}: {exc}") from exc

    triggers = data.get("triggers", [])
    if trigger_index < 0 or trigger_index >= len(triggers):
        raise IndexError("Выбранный триггер не существует.")

    images = triggers[trigger_index].get("images", [])
    if image_index < 0 or image_index >= len(images):
        raise IndexError("Выбранной картинки нет в триггере.")

    removed = images.pop(image_index)
    preset_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return removed


def build_unique_destination(preset_name: str, extension: str, *, base_suffix: Optional[str] = None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = sanitize_suffix(base_suffix) if base_suffix else ""
    base_name = f"{preset_name}_{timestamp}{f'_{suffix}' if suffix else ''}"
    destination = PRESETS_DIR / f"{base_name}{extension}"
    counter = 1
    while destination.exists():
        destination = PRESETS_DIR / f"{base_name}_{counter}{extension}"
        counter += 1
    return destination


def sanitize_suffix(value: Optional[str]) -> str:
    if not value:
        return ""
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    cleaned = cleaned.strip("_")
    return cleaned or "file"


def compute_file_hash_from_path(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_existing_image_hashes() -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    if not PRESETS_DIR.exists():
        return hashes
    for file_path in PRESETS_DIR.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS:
            try:
                hashes[compute_file_hash_from_path(file_path)] = file_path.name
            except OSError as exc:
                logger.warning("Не удалось вычислить хеш для %s: %s", file_path, exc)
    return hashes


async def save_single_image(
    telegram_file,
    preset_name: str,
    extension: str,
    base_suffix: Optional[str],
    existing_hashes: Dict[str, str],
) -> Tuple[List[str], List[str]]:
    saved_files: List[str] = []
    skipped: List[str] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir) / "upload"
        await telegram_file.download_to_drive(custom_path=str(temp_path))
        file_hash = compute_file_hash_from_path(temp_path)
        if file_hash in existing_hashes:
            skipped.append(existing_hashes[file_hash])
            return saved_files, skipped

        destination = build_unique_destination(preset_name, extension, base_suffix=base_suffix)
        shutil.move(str(temp_path), destination)
        saved_files.append(destination.name)
        existing_hashes[file_hash] = destination.name
    return saved_files, skipped


async def save_images_from_zip(
    telegram_file,
    preset_name: str,
    existing_hashes: Dict[str, str],
) -> Tuple[List[str], List[str]]:
    saved_files: List[str] = []
    skipped: List[str] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        archive_path = Path(tmp_dir) / "upload.zip"
        await telegram_file.download_to_drive(custom_path=str(archive_path))
        try:
            with zipfile.ZipFile(archive_path) as archive:
                for index, info in enumerate(archive.infolist(), start=1):
                    if info.is_dir():
                        continue
                    inner_path = Path(info.filename)
                    extension = inner_path.suffix.lower()
                    if extension not in ALLOWED_IMAGE_EXTENSIONS:
                        continue
                    suffix = f"zip_{index}_{inner_path.stem[:20]}"
                    temp_file = Path(tmp_dir) / f"entry_{index}"
                    digest = sha256()
                    with archive.open(info) as source, temp_file.open("wb") as target:
                        while True:
                            chunk = source.read(65536)
                            if not chunk:
                                break
                            digest.update(chunk)
                            target.write(chunk)
                    file_hash = digest.hexdigest()
                    if file_hash in existing_hashes:
                        skipped.append(existing_hashes[file_hash])
                        temp_file.unlink(missing_ok=True)
                        continue
                    destination = build_unique_destination(
                        preset_name,
                        extension,
                        base_suffix=suffix,
                    )
                    shutil.move(str(temp_file), destination)
                    saved_files.append(destination.name)
                    existing_hashes[file_hash] = destination.name
        except zipfile.BadZipFile as exc:
            raise ValueError("Не удалось распаковать ZIP-архив. Проверьте файл.") from exc

    return saved_files, skipped


def admin_panel_text() -> str:
    return (
        "Панель администрирования:\n"
        "Выберите действие из списка или используйте /-команды."
    )


def build_admin_main_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("ℹ️ /start", callback_data="admin_panel:start")],
            [InlineKeyboardButton("❓ /help", callback_data="admin_panel:help")],
            [InlineKeyboardButton("📋 /list_presets", callback_data="admin_panel:list_presets")],
            [InlineKeyboardButton("🔄 /reload_presets", callback_data="admin_panel:reload_presets")],
            [InlineKeyboardButton("📌 /apply_preset", callback_data="admin_panel:apply_preset")],
            [InlineKeyboardButton("📂 /show_state", callback_data="admin_panel:show_state")],
            [InlineKeyboardButton("🛠 /debug_status", callback_data="admin_panel:debug_status")],
            [InlineKeyboardButton("💬 /debug_chat", callback_data="admin_panel:debug_chat")],
            [InlineKeyboardButton("🧩 /list_triggers", callback_data="admin_panel:list_triggers")],
            [InlineKeyboardButton("🖼 Управление картинками", callback_data="admin_panel:images")],
        ]
    )


def build_back_to_admin_panel_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton("🔙 В главное меню", callback_data="admin_panel:menu")]])


def build_admin_images_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("🖼 Загрузить пресет", callback_data="admin_presets:upload")],
            [InlineKeyboardButton("📋 Список пресетов", callback_data="admin_presets:list")],
            [InlineKeyboardButton("📷 Просмотреть картинки", callback_data="admin_presets:view_images")],
            [InlineKeyboardButton("✏️ Переименовать триггер", callback_data="admin_presets:edit_trigger")],
            [InlineKeyboardButton("🗑 Удалить картинку", callback_data="admin_presets:delete_image")],
            [InlineKeyboardButton("🗑 Удалить пресет", callback_data="admin_presets:delete")],
            [InlineKeyboardButton("🔙 В админ-панель", callback_data="admin_presets:exit")],
        ]
    )


def build_back_to_presets_menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Назад", callback_data="admin_presets:menu")]])


def build_presets_selection_keyboard(action: str, preset_names: List[str]) -> InlineKeyboardMarkup:
    buttons = [[InlineKeyboardButton(name, callback_data=f"{action}:{name}")] for name in preset_names]
    buttons.append([InlineKeyboardButton("🔙 Назад", callback_data="admin_presets:menu")])
    return InlineKeyboardMarkup(buttons)


def build_presets_delete_keyboard(preset_names: List[str]) -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton(f"🗑 {name}", callback_data=f"admin_presets:delete_confirm:{name}")]
        for name in preset_names
    ]
    buttons.append([InlineKeyboardButton("🔙 Назад", callback_data="admin_presets:menu")])
    return InlineKeyboardMarkup(buttons)


def build_image_files_keyboard(action: str, file_names: List[str]) -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton(f"🖼 {name}", callback_data=f"{action}:{index}")]
        for index, name in enumerate(file_names)
    ]
    buttons.append([InlineKeyboardButton("🔙 Назад", callback_data="admin_presets:menu")])
    return InlineKeyboardMarkup(buttons)


def build_triggers_selection_keyboard(
    preset_name: str,
    preset: Dict[str, Any],
    *,
    action_prefix: str = "upload_trigger",
    back_callback: str = "admin_presets:upload",
) -> InlineKeyboardMarkup:
    trigger_buttons: List[List[InlineKeyboardButton]] = []
    for index, trigger in enumerate(preset.get("triggers", [])):
        phrase = str(trigger.get("phrase") or "").strip()
        keywords = [str(keyword).strip() for keyword in trigger.get("keywords", []) if str(keyword).strip()]
        label = phrase or (", ".join(keywords[:3]) + ("…" if len(keywords) > 3 else ""))
        if not label:
            label = f"Триггер {index + 1}"
        trigger_buttons.append(
            [
                InlineKeyboardButton(
                    f"{index + 1}. {label}",
                    callback_data=f"admin_presets:{action_prefix}:{preset_name}:{index}",
                )
            ]
        )

    if not trigger_buttons:
        trigger_buttons.append([InlineKeyboardButton("Нет триггеров", callback_data="admin_presets:menu")])

    trigger_buttons.append([InlineKeyboardButton("🔙 Назад", callback_data=back_callback)])
    return InlineKeyboardMarkup(trigger_buttons)


def build_trigger_images_keyboard(
    preset_name: str,
    trigger_index: int,
    images: List[str],
    *,
    action_prefix: str,
    back_callback: str,
) -> InlineKeyboardMarkup:
    buttons = [
        [
            InlineKeyboardButton(
                f"🗑 {image_name}",
                callback_data=f"admin_presets:{action_prefix}:{preset_name}:{trigger_index}:{index}",
            )
        ]
        for index, image_name in enumerate(images)
    ]
    buttons.append([InlineKeyboardButton("🔙 Назад", callback_data=back_callback)])
    return InlineKeyboardMarkup(buttons)

def list_preset_files() -> List[Path]:
    ensure_presets_directory()
    return sorted(
        file_path
        for file_path in PRESETS_DIR.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
    )

def list_image_files() -> List[Path]:
    """Deprecated helper preserved for backward compatibility."""
    return list_preset_files()


def image_is_referenced(image_name: str, presets: Dict[str, Dict[str, Any]]) -> bool:
    for preset in presets.values():
        for trigger in preset.get("triggers", []):
            images = [str(ref) for ref in trigger.get("images", []) if isinstance(ref, str)]
            if image_name in images:
                return True
    return False

async def show_admin_main_menu(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    edit: bool = False,
    answer_callback: bool = True,
) -> None:
    keyboard = build_admin_main_keyboard()
    text = admin_panel_text()
    context.user_data.pop("admin_panel_pending", None)

    if edit and update.callback_query and update.callback_query.message:
        await update.callback_query.message.edit_text(text, reply_markup=keyboard)
        context.user_data["admin_panel_message_id"] = update.callback_query.message.message_id
        if answer_callback:
            await update.callback_query.answer()
    else:
        sent = await update.effective_message.reply_text(text, reply_markup=keyboard)
        context.user_data["admin_panel_message_id"] = sent.message_id


async def restore_admin_main_menu_from_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    message_id = context.user_data.get("admin_panel_message_id")
    if chat and message_id:
        try:
            await context.bot.edit_message_text(
                admin_panel_text(),
                chat_id=chat.id,
                message_id=message_id,
                reply_markup=build_admin_main_keyboard(),
            )
            context.user_data["admin_panel_message_id"] = message_id
            context.user_data.pop("admin_panel_pending", None)
            return
        except Exception as exc:
            logger.debug("Failed to edit admin panel message %s: %s", message_id, exc)

    sent = await update.effective_message.reply_text(admin_panel_text(), reply_markup=build_admin_main_keyboard())
    context.user_data["admin_panel_message_id"] = sent.message_id


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

    logger.info("Admin %s opened admin panel.", update.effective_user.id if update.effective_user else "unknown")
    await show_admin_main_menu(update, context)


async def invoke_command(handler, update: Update, context: ContextTypes.DEFAULT_TYPE, args: Optional[List[str]] = None) -> None:
    current_args = getattr(context, "args", None)
    previous_args = list(current_args) if current_args else []
    try:
        context.args = args or []
        await handler(update, context)
    finally:
        context.args = previous_args


async def handle_admin_panel_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await ensure_admin_panel_access(update):
        return

    query = update.callback_query
    if not query:
        return

    data = query.data or ""
    parts = data.split(":", 1)
    if len(parts) != 2 or parts[0] != "admin_panel":
        await query.answer()
        return

    action = parts[1]
    logger.debug("Admin %s triggered admin panel action %s.", update.effective_user.id if update.effective_user else "unknown", action)

    if action == "menu":
        await show_admin_main_menu(update, context, edit=True, answer_callback=False)
        await query.answer("Главное меню обновлено.")
        return

    if action == "images":
        context.user_data.pop("admin_panel_pending", None)
        await show_admin_presets_menu(update, context, edit=True)
        return

    simple_command_handlers = {
        "start": cmd_start,
        "help": cmd_help,
        "list_presets": cmd_list_presets,
        "reload_presets": cmd_reload_presets,
        "show_state": cmd_show_state,
        "debug_status": cmd_debug_status,
    }

    if action in simple_command_handlers:
        await invoke_command(simple_command_handlers[action], update, context)
        await show_admin_main_menu(update, context, edit=True, answer_callback=False)
        await query.answer("Команда выполнена.")
        return

    if action == "list_triggers":
        context.user_data["admin_panel_pending"] = {"action": "list_triggers"}
        await query.message.edit_text(
            "Введите имя пресета, триггеры которого хотите посмотреть:",
            reply_markup=build_back_to_admin_panel_keyboard(),
        )
        context.user_data["admin_panel_message_id"] = query.message.message_id
        await query.answer("Ожидаю имя пресета.")
        return

    if action == "apply_preset":
        context.user_data["admin_panel_pending"] = {"action": "apply_preset"}
        await query.message.edit_text(
            "Введите chat_id и имя пресета через пробел.\nПример: -100123456789 anya_redko",
            reply_markup=build_back_to_admin_panel_keyboard(),
        )
        context.user_data["admin_panel_message_id"] = query.message.message_id
        await query.answer("Ожидаю параметры для назначения пресета.")
        return

    if action == "debug_chat":
        context.user_data["admin_panel_pending"] = {"action": "debug_chat"}
        await query.message.edit_text(
            "Введите chat_id, по которому нужна диагностика:",
            reply_markup=build_back_to_admin_panel_keyboard(),
        )
        context.user_data["admin_panel_message_id"] = query.message.message_id
        await query.answer("Ожидаю chat_id.")
        return

    await query.answer()


async def handle_admin_presets_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await ensure_admin_panel_access(update):
        return

    query = update.callback_query
    if not query:
        return

    data = query.data or ""
    logger.debug("Admin %s triggered callback %s.", update.effective_user.id if update.effective_user else "unknown", data)

    if data == "admin_presets:menu":
        await show_admin_presets_menu(update, context, edit=True)
        return

    if data == "admin_presets:exit":
        await show_admin_main_menu(update, context, edit=True)
        return

    if data == "admin_presets:upload":
        presets = context.bot_data.setdefault("presets", {})
        context.user_data.pop("awaiting_preset_upload", None)
        if not presets:
            await query.message.edit_text(
                "Сначала добавьте JSON-пресеты и выполните /obnovit_presets.",
                reply_markup=build_back_to_presets_menu_keyboard(),
            )
            await query.answer()
            return

        keyboard = build_presets_selection_keyboard("admin_presets:upload_preset", sorted(presets.keys()))
        await query.message.edit_text(
            "Выберите пресет, к которому нужно добавить картинку:",
            reply_markup=keyboard,
        )
        await query.answer()
        return

    if data.startswith("admin_presets:upload_preset:"):
        _, _, preset_name = data.split(":", 2)
        presets = context.bot_data.setdefault("presets", {})
        preset = presets.get(preset_name)
        if not preset:
            await query.answer("Пресет не найден.", show_alert=True)
            return

        if not preset.get("triggers"):
            await query.message.edit_text(
                "В этом пресете нет триггеров. Добавьте их вручную в JSON-файл.",
                reply_markup=build_back_to_presets_menu_keyboard(),
            )
            await query.answer()
            return

        keyboard = build_triggers_selection_keyboard(preset_name, preset)
        await query.message.edit_text(
            f"Выбран пресет {preset_name}. Теперь выберите триггер:",
            reply_markup=keyboard,
        )
        await query.answer()
        return

    if data.startswith("admin_presets:upload_trigger:"):
        _, _, preset_name, trigger_index_str = data.split(":", 3)
        presets = context.bot_data.setdefault("presets", {})
        preset = presets.get(preset_name)
        if not preset:
            await query.answer("Пресет не найден.", show_alert=True)
            return

        try:
            trigger_index = int(trigger_index_str)
        except ValueError:
            await query.answer("Некорректный номер триггера.", show_alert=True)
            return

        triggers = preset.get("triggers", [])
        if trigger_index < 0 or trigger_index >= len(triggers):
            await query.answer("Такого триггера нет.", show_alert=True)
            return

        context.user_data["awaiting_preset_upload"] = {
            "preset_name": preset_name,
            "trigger_index": trigger_index,
        }
        await query.message.edit_text(
            f"Отправьте изображение для добавления в пресет «{preset_name}».\n"
            "Можно прислать фото или файл формата JPG/PNG/WebP.",
            reply_markup=build_back_to_presets_menu_keyboard(),
        )
        await query.answer()
        return

    if data == "admin_presets:list":
        presets = context.bot_data.setdefault("presets", {})
        if not presets:
            await query.message.edit_text(
                "Пресеты не найдены. Добавьте JSON-файлы и выполните /obnovit_presets.",
                reply_markup=build_back_to_presets_menu_keyboard(),
            )
            await query.answer()
            return

        lines = []
        for preset_name, preset in presets.items():
            description = preset.get("description")
            lines.append(f"- {preset_name}" + (f" ({description})" if description else ""))

        await query.message.edit_text(
            "Доступные пресеты:\n" + "\n".join(lines),
            reply_markup=build_back_to_presets_menu_keyboard(),
        )
        await query.answer()
        return

    if data == "admin_presets:view_images":
        files = list_preset_files()
        if not files:
            await query.message.edit_text(
                "В папке presets пока нет изображений.",
                reply_markup=build_back_to_presets_menu_keyboard(),
            )
            await query.answer()
            return

        file_names = [file_path.name for file_path in files]
        context.user_data["image_gallery"] = file_names

        await query.message.edit_text(
            "Выберите картинку для просмотра:",
            reply_markup=build_image_files_keyboard("admin_presets:view_file_idx", file_names),
        )
        await query.answer()
        return

    if data == "admin_presets:edit_trigger":
        presets = context.bot_data.setdefault("presets", {})
        if not presets:
            await query.message.edit_text(
                "Пресеты не найдены. Добавьте JSON-файлы и выполните /obnovit_presets.",
                reply_markup=build_back_to_presets_menu_keyboard(),
            )
            await query.answer()
            return

        keyboard = build_presets_selection_keyboard("admin_presets:edit_preset", sorted(presets.keys()))
        await query.message.edit_text(
            "Выберите пресет, где хотите изменить фразу-триггер:",
            reply_markup=keyboard,
        )
        await query.answer()
        return

    if data.startswith("admin_presets:edit_preset:"):
        _, _, preset_name = data.split(":", 2)
        presets = context.bot_data.setdefault("presets", {})
        preset = presets.get(preset_name)
        if not preset:
            await query.answer("Пресет не найден.", show_alert=True)
            return

        keyboard = build_triggers_selection_keyboard(
            preset_name,
            preset,
            action_prefix="edit_trigger_select",
            back_callback="admin_presets:edit_trigger",
        )
        await query.message.edit_text(
            f"Пресет {preset_name}. Выберите триггер для изменения:",
            reply_markup=keyboard,
        )
        await query.answer()
        return

    if data.startswith("admin_presets:edit_trigger_select:"):
        parts = data.split(":", 3)
        if len(parts) != 4:
            await query.answer("Некорректный параметр.", show_alert=True)
            return
        _, _, preset_name, trigger_index_str = parts
        presets = context.bot_data.setdefault("presets", {})
        preset = presets.get(preset_name)
        if not preset:
            await query.answer("Пресет не найден.", show_alert=True)
            return

        try:
            trigger_index = int(trigger_index_str)
        except ValueError:
            await query.answer("Некорректный номер триггера.", show_alert=True)
            return

        triggers = preset.get("triggers", [])
        if trigger_index < 0 or trigger_index >= len(triggers):
            await query.answer("Такого триггера нет.", show_alert=True)
            return

        context.user_data["admin_panel_pending"] = {
            "action": "edit_trigger_phrase",
            "preset_name": preset_name,
            "trigger_index": trigger_index,
            "return_to": "admin_presets",
        }
        await query.message.edit_text(
            f"Введите новую фразу для триггера №{trigger_index + 1} пресета «{preset_name}»:",
            reply_markup=build_back_to_presets_menu_keyboard(),
        )
        context.user_data["admin_panel_message_id"] = query.message.message_id
        await query.answer("Ожидаю новую фразу.")
        return

    if data == "admin_presets:delete_image":
        presets = context.bot_data.setdefault("presets", {})
        if not presets:
            await query.message.edit_text(
                "Пресеты не найдены. Добавьте JSON-файлы и выполните /obnovit_presets.",
                reply_markup=build_back_to_presets_menu_keyboard(),
            )
            await query.answer()
            return

        keyboard = build_presets_selection_keyboard("admin_presets:delete_image_preset", sorted(presets.keys()))
        await query.message.edit_text(
            "Выберите пресет, из которого удаляем изображения:",
            reply_markup=keyboard,
        )
        await query.answer()
        return

    if data.startswith("admin_presets:delete_image_preset:"):
        _, _, preset_name = data.split(":", 2)
        presets = context.bot_data.setdefault("presets", {})
        preset = presets.get(preset_name)
        if not preset:
            await query.answer("Пресет не найден.", show_alert=True)
            return

        keyboard = build_triggers_selection_keyboard(
            preset_name,
            preset,
            action_prefix="delete_image_trigger",
            back_callback="admin_presets:delete_image",
        )
        await query.message.edit_text(
            f"Пресет {preset_name}. Выберите триггер, где нужно удалить картинку:",
            reply_markup=keyboard,
        )
        await query.answer()
        return

    if data.startswith("admin_presets:delete_image_trigger:"):
        parts = data.split(":", 3)
        if len(parts) != 4:
            await query.answer("Некорректный параметр.", show_alert=True)
            return
        _, _, preset_name, trigger_index_str = parts
        presets = context.bot_data.setdefault("presets", {})
        preset = presets.get(preset_name)
        if not preset:
            await query.answer("Пресет не найден.", show_alert=True)
            return

        try:
            trigger_index = int(trigger_index_str)
        except ValueError:
            await query.answer("Некорректный номер триггера.", show_alert=True)
            return

        triggers = preset.get("triggers", [])
        if trigger_index < 0 or trigger_index >= len(triggers):
            await query.answer("Такого триггера нет.", show_alert=True)
            return

        images = [str(image) for image in triggers[trigger_index].get("images", []) if isinstance(image, str)]
        if not images:
            await query.message.edit_text(
                "В выбранном триггере нет картинок.",
                reply_markup=build_triggers_selection_keyboard(
                    preset_name,
                    preset,
                    action_prefix="delete_image_trigger",
                    back_callback="admin_presets:delete_image",
                ),
            )
            await query.answer()
            return

        keyboard = build_trigger_images_keyboard(
            preset_name,
            trigger_index,
            images,
            action_prefix="delete_image_file",
            back_callback=f"admin_presets:delete_image_preset:{preset_name}",
        )
        await query.message.edit_text(
            "Выберите картинку для удаления:",
            reply_markup=keyboard,
        )
        await query.answer()
        return

    if data.startswith("admin_presets:delete_image_file:"):
        parts = data.split(":", 5)
        if len(parts) != 5:
            await query.answer("Некорректный параметр.", show_alert=True)
            return
        _, _, preset_name, trigger_index_str, image_index_str = parts
        presets = context.bot_data.setdefault("presets", {})
        preset = presets.get(preset_name)
        if not preset:
            await query.answer("Пресет не найден.", show_alert=True)
            return

        try:
            trigger_index = int(trigger_index_str)
            image_index = int(image_index_str)
        except ValueError:
            await query.answer("Некорректный индекс.", show_alert=True)
            return

        try:
            removed_image = remove_image_from_trigger(preset_name, trigger_index, image_index)
        except Exception as exc:
            await query.answer(str(exc), show_alert=True)
            return

        presets = load_presets()
        context.bot_data["presets"] = presets
        if not image_is_referenced(removed_image, presets):
            (PRESETS_DIR / removed_image).unlink(missing_ok=True)

        preset = presets.get(preset_name)
        trigger = preset.get("triggers", [])[trigger_index] if preset and trigger_index < len(preset.get("triggers", [])) else {}
        images = [str(image) for image in trigger.get("images", []) if isinstance(image, str)]
        if images:
            keyboard = build_trigger_images_keyboard(
                preset_name,
                trigger_index,
                images,
                action_prefix="delete_image_file",
                back_callback=f"admin_presets:delete_image_preset:{preset_name}",
            )
            await query.message.edit_text(
                f"Удалено: {removed_image}\nВыберите следующую картинку для удаления:",
                reply_markup=keyboard,
            )
        else:
            keyboard = build_triggers_selection_keyboard(
                preset_name,
                preset or {},
                action_prefix="delete_image_trigger",
                back_callback="admin_presets:delete_image",
            )
            await query.message.edit_text(
                f"Удалено: {removed_image}\nВ этом триггере больше нет картинок.",
                reply_markup=keyboard,
            )
        await query.answer("Картинка удалена.")
        return

    if data.startswith("admin_presets:view_file_idx:"):
        parts = data.split(":", 3)
        if len(parts) < 3:
            await query.answer("Некорректный параметр.", show_alert=True)
            return
        try:
            index = int(parts[2])
        except ValueError:
            await query.answer("Некорректный индекс.", show_alert=True)
            return

        gallery = context.user_data.get("image_gallery", [])
        if not gallery or index < 0 or index >= len(gallery):
            await query.answer("Не удалось найти изображение.", show_alert=True)
            return

        filename = gallery[index]
        file_path = resolve_image_path(filename)
        if not file_path:
            await query.answer("Файл не найден.", show_alert=True)
            return

        with file_path.open("rb") as image_file:
            await query.message.reply_photo(photo=image_file, caption=filename)
        logger.info(
            "Admin %s viewed preset image %s.",
            update.effective_user.id if update.effective_user else "unknown",
            file_path,
        )
        await query.answer()
        return

    if data == "admin_presets:delete":
        presets = context.bot_data.setdefault("presets", {})
        if not presets:
            await query.message.edit_text(
                "Нет пресетов для удаления.",
                reply_markup=build_back_to_presets_menu_keyboard(),
            )
            await query.answer()
            return

        await query.message.edit_text(
            "Выберите пресет для удаления:",
            reply_markup=build_presets_delete_keyboard(sorted(presets.keys())),
        )
        await query.answer()
        return

    if data.startswith("admin_presets:delete_confirm:"):
        _, _, preset_name = data.split(":", 2)
        preset_path = get_preset_file_path(preset_name)
        if not preset_path:
            await query.answer("Файл пресета не найден.", show_alert=True)
            return

        try:
            preset_path.unlink()
        except OSError as exc:
            await query.answer(f"Не удалось удалить пресет: {exc}", show_alert=True)
            return

        presets = load_presets()
        context.bot_data["presets"] = presets

        state = context.bot_data.setdefault("state", {})
        removed_chats = [chat_id for chat_id, assigned in state.items() if assigned == preset_name]
        for chat_id in removed_chats:
            del state[chat_id]
        if removed_chats:
            save_state(state)

        await query.message.edit_text(
            f"🗑 Пресет {preset_name} удалён."
            + (f" Удалены назначения для чатов: {', '.join(removed_chats)}" if removed_chats else ""),
            reply_markup=build_admin_images_keyboard(),
        )
        logger.info(
            "Admin %s deleted preset %s.",
            update.effective_user.id if update.effective_user else "unknown",
            preset_name,
        )
        await query.answer()
        return

    await query.answer()


async def handle_admin_image_upload(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    upload_context = context.user_data.get("awaiting_preset_upload")
    if not upload_context:
        return

    if not await ensure_admin_panel_access(update):
        return

    message = update.effective_message
    if not message:
        return

    ensure_presets_directory()
    existing_hashes = build_existing_image_hashes()
    preset_name = upload_context.get("preset_name")
    trigger_index = upload_context.get("trigger_index")
    if preset_name is None or trigger_index is None:
        context.user_data.pop("awaiting_preset_upload", None)
        await message.reply_text("Возникла ошибка при выборе пресета. Попробуйте начать заново.")
        return

    telegram_file = None
    is_archive = False
    extension = ".jpg"
    base_suffix: Optional[str] = None

    if message.photo:
        telegram_file = await message.photo[-1].get_file()
    elif message.document:
        doc = message.document
        telegram_file = await doc.get_file()
        file_name = doc.file_name or ""
        base_suffix = Path(file_name).stem if file_name else None
        ext = Path(file_name).suffix.lower()
        mime = (doc.mime_type or "").lower()
        if ext in ALLOWED_ARCHIVE_EXTENSIONS or "zip" in mime:
            is_archive = True
        elif ext in ALLOWED_IMAGE_EXTENSIONS:
            extension = ext
        elif telegram_file and telegram_file.file_path:
            _, inferred_ext = os.path.splitext(telegram_file.file_path)
            inferred_ext = inferred_ext.lower()
            if inferred_ext in ALLOWED_IMAGE_EXTENSIONS:
                extension = inferred_ext
    elif message.document is None and message.photo is None:
        telegram_file = None

    if not telegram_file:
        await message.reply_text("⚠️ Пожалуйста, отправьте изображение (фото или файл) или ZIP-архив с изображениями.")
        return

    saved_filenames: List[str] = []
    skipped_duplicates: List[str] = []
    if is_archive:
        try:
            saved_filenames, skipped_duplicates = await save_images_from_zip(telegram_file, preset_name, existing_hashes)
        except ValueError as exc:
            await message.reply_text(f"❌ {exc}")
            return

        if not saved_filenames:
            if skipped_duplicates:
                await message.reply_text("⚠️ Все изображения из архива уже есть среди загруженных и были пропущены.")
            else:
                await message.reply_text("⚠️ В архиве не найдено подходящих изображений (JPG/PNG/WebP/GIF).")
            return
    else:
        saved_filenames, skipped_duplicates = await save_single_image(
            telegram_file,
            preset_name,
            extension,
            base_suffix,
            existing_hashes,
        )
        if not saved_filenames:
            await message.reply_text("⚠️ Это изображение уже загружено ранее и было пропущено.")
            return

    try:
        append_image_references_to_preset(preset_name, int(trigger_index), saved_filenames)
        context.bot_data["presets"] = load_presets()
    except Exception as exc:
        for filename in saved_filenames:
            (PRESETS_DIR / filename).unlink(missing_ok=True)
        logger.exception("Failed to attach uploaded image(s) to preset %s: %s", preset_name, exc)
        await message.reply_text(f"❌ Не удалось обновить пресет: {exc}")
        return

    context.user_data.pop("awaiting_preset_upload", None)

    logger.info(
        "Admin %s uploaded %d preset image(s) %s.",
        update.effective_user.id if update.effective_user else "unknown",
        len(saved_filenames),
        saved_filenames,
    )

    duplicate_note = ""
    if skipped_duplicates:
        duplicate_note = f"\nПропущено дубликатов: {len(skipped_duplicates)}"

    await message.reply_text(
        "✅ Картинки загружены и привязаны к выбранному триггеру.\n"
        f"Файлы: {', '.join(saved_filenames)}{duplicate_note}"
    )
    await message.reply_text(
        "Возвращаю вас в меню админки:",
        reply_markup=build_admin_images_keyboard(),
    )


async def handle_admin_panel_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    pending = context.user_data.get("admin_panel_pending")
    if not pending:
        return

    if not await ensure_admin_panel_access(update):
        return

    message = update.effective_message
    if not message or not message.text:
        return

    text = message.text.strip()
    if not text:
        await message.reply_text("Введите текстовое значение.")
        return

    action = pending.get("action")
    restore_main_menu = True
    if action == "apply_preset":
        parts = text.split(maxsplit=1)
        if len(parts) != 2:
            await message.reply_text("Нужно указать chat_id и имя пресета через пробел.")
            return
        await invoke_command(cmd_apply_preset, update, context, args=parts)
    elif action == "debug_chat":
        await invoke_command(cmd_debug_chat, update, context, args=[text])
    elif action == "list_triggers":
        await invoke_command(cmd_list_triggers, update, context, args=[text])
    elif action == "edit_trigger_phrase":
        preset_name = pending.get("preset_name")
        trigger_index = pending.get("trigger_index")
        if preset_name is None or trigger_index is None:
            await message.reply_text("Не удалось определить пресет или триггер. Попробуйте заново.")
            return

        try:
            update_trigger_phrase_in_preset(preset_name, int(trigger_index), text)
        except Exception as exc:
            await message.reply_text(f"Не удалось обновить триггер: {exc}")
            return

        context.bot_data["presets"] = load_presets()
        await message.reply_text(
            f"✅ Триггер пресета «{preset_name}» обновлён.\nНовая фраза: {text}"
        )
        if pending.get("return_to") == "admin_presets":
            restore_main_menu = False
            await message.reply_text(
                "Возвращаю вас в меню управления пресетами:",
                reply_markup=build_admin_images_keyboard(),
            )
    else:
        return

    context.user_data.pop("admin_panel_pending", None)
    if restore_main_menu:
        await restore_admin_main_menu_from_text(update, context)


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
    application.add_handler(CallbackQueryHandler(handle_admin_panel_callback, pattern=r"^admin_panel:"))
    application.add_handler(CallbackQueryHandler(handle_admin_presets_callback, pattern=r"^admin_presets:"))
    application.add_handler(
        MessageHandler(
            filters.ChatType.PRIVATE & (filters.PHOTO | filters.Document.ALL),
            handle_admin_image_upload,
        )
    )
    application.add_handler(
        MessageHandler(
            filters.ChatType.PRIVATE & filters.TEXT & (~filters.COMMAND),
            handle_admin_panel_text,
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
