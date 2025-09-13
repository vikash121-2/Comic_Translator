import os
import shutil
import requests
import zipfile
import datetime
from collections import defaultdict
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ChatAction
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters

# OCR / image
import easyocr
from PIL import Image, ImageDraw, ImageFont
from deep_translator import GoogleTranslator
import re

# -------------------------
# Configuration & Globals
# -------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
FONT_DIR = os.path.join(ROOT_DIR, 'fonts')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FONT_DIR, exist_ok=True)

ALLOWED_IMAGE_EXTS = {"png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"}
TARGET_LANGUAGE = 'english'  # minimal: translate everything to English

# Utility: safe filename without werkzeug
_filename_strip_re = re.compile(r'[^A-Za-z0-9_.-]')

def secure_filename(name: str) -> str:
    if not name:
        return ""
    name = name.strip().replace(' ', '_')
    name = _filename_strip_re.sub('', name)
    # avoid path separators
    name = name.replace('..', '').replace('/', '').replace('\\', '')
    return name or 'file'

# Download a CJK font once for better rendering
FONT_PATH = os.path.join(FONT_DIR, "NotoSansCJK.otf")
if not os.path.exists(FONT_PATH):
    try:
        print("Downloading CJK font...")
        font_url = "https://cdn.jsdelivr.net/gh/googlefonts/noto-cjk@main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf"
        response = requests.get(font_url, stream=True, timeout=60)
        if response.ok:
            with open(FONT_PATH, 'wb') as f:
                f.write(response.content)
            print("Font downloaded.")
        else:
            print(f"Failed to download font: {response.status_code}")
            FONT_PATH = None
    except Exception as e:
        print(f"Font download error: {e}")
        FONT_PATH = None

# EasyOCR reader (lazy init)
reader = None

def initialize_ocr():
    global reader
    if reader is None:
        try:
            print("Initializing EasyOCR reader...")
            reader = easyocr.Reader(['en', 'ja', 'ch_sim'], gpu=False)
            print("EasyOCR initialized.")
        except Exception as e:
            print(f"EasyOCR init failed: {e}")
            reader = None

# -------------------------
# Helpers
# -------------------------

def get_user_dirs(user_id: int):
    base = os.path.join(DATA_DIR, str(user_id))
    uploads = os.path.join(base, 'uploads')
    translated = os.path.join(base, 'translated')
    archives = os.path.join(base, 'archives')
    for d in (base, uploads, translated, archives):
        os.makedirs(d, exist_ok=True)
    return base, uploads, translated, archives


def cleanup_user(user_id: int):
    base = os.path.join(DATA_DIR, str(user_id))
    if os.path.exists(base):
        shutil.rmtree(base, ignore_errors=True)


def allowed_image_file(filename: str) -> bool:
    if not filename:
        return False
    ext = os.path.splitext(filename)[1].lower().lstrip('.')
    return ext in ALLOWED_IMAGE_EXTS


def allowed_zip_file(filename: str) -> bool:
    return filename and os.path.splitext(filename)[1].lower() == '.zip'


def safe_extract(zip_path: str, extract_to: str):
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            member_name = member.filename
            dest_path = os.path.abspath(os.path.join(extract_to, member_name))
            if not dest_path.startswith(os.path.abspath(extract_to) + os.sep):
                continue
            if member.is_dir():
                os.makedirs(dest_path, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                with zf.open(member) as src, open(dest_path, 'wb') as out:
                    shutil.copyfileobj(src, out)


def get_optimal_font(draw, text, max_width, max_height):
    font_size = max(12, int(max_height))
    font = ImageFont.load_default()
    if FONT_PATH and os.path.exists(FONT_PATH):
        try:
            font = ImageFont.truetype(FONT_PATH, font_size)
        except Exception:
            pass
    while font_size > 6:
        if hasattr(draw, 'textbbox'):
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        else:
            text_width, text_height = draw.textsize(text, font=font)
        if text_width <= max_width and text_height <= max_height:
            break
        font_size -= 1
        if FONT_PATH and os.path.exists(FONT_PATH):
            try:
                font = ImageFont.truetype(FONT_PATH, font_size)
            except Exception:
                font = ImageFont.load_default()
    return font


def translate_text(text: str, target_language: str) -> str:
    if not text or not text.strip():
        return ""
    try:
        lang_map = {'chinese (simplified)': 'zh-CN'}
        target_code = lang_map.get(target_language.lower(), target_language)
        return GoogleTranslator(source='auto', target=target_code).translate(text)
    except Exception as e:
        print(f"Translation failed for '{text}': {e}")
        return text


def process_image(original_full_path: str, relative_filepath: str, target_language: str, translated_root: str):
    translated_full_path = os.path.join(translated_root, relative_filepath)
    os.makedirs(os.path.dirname(translated_full_path), exist_ok=True)
    original_texts = []
    if not reader:
        shutil.copy(original_full_path, translated_full_path)
        return relative_filepath, original_texts
    try:
        result = reader.readtext(original_full_path, paragraph=False)
        if not result:
            shutil.copy(original_full_path, translated_full_path)
            return relative_filepath, original_texts
        image = Image.open(original_full_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        for (bbox, text, prob) in result:
            draw.polygon([tuple(p) for p in bbox], fill='white')
        for (bbox, text, prob) in result:
            original_texts.append(text)
            translated_text = translate_text(text, target_language)
            x_coords, y_coords = [p[0] for p in bbox], [p[1] for p in bbox]
            top_left = (min(x_coords), min(y_coords))
            bottom_right = (max(x_coords), max(y_coords))
            box_width = max(1, bottom_right[0] - top_left[0])
            box_height = max(1, bottom_right[1] - top_left[1])
            font = get_optimal_font(draw, translated_text, box_width, box_height)
            if hasattr(draw, 'textbbox'):
                text_bbox = draw.textbbox((0, 0), translated_text, font=font)
                text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            else:
                text_width, text_height = draw.textsize(translated_text, font=font)
            new_x = top_left[0] + (box_width - text_width) / 2
            new_y = top_left[1] + (box_height - text_height) / 2
            draw.text((new_x, new_y), translated_text, font=font, fill='black', stroke_width=2, stroke_fill='white')
        image.save(translated_full_path)
    except Exception as e:
        print(f"Image processing failed: {e}")
        shutil.copy(original_full_path, translated_full_path)
    return relative_filepath, original_texts


def create_text_file(text_structure, base_name: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    filename = f"original_text_{secure_filename(base_name)}.txt"
    filepath = os.path.join(out_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        for chapter, pages in sorted(text_structure.items()):
            f.write(f"--- Chapter: {chapter} ---\n\n")
            for page, texts in sorted(pages.items()):
                f.write(f"--- Image: {page} ---\n")
                for text in texts:
                    f.write(f"{text}\n")
                f.write("\n")
    return filepath


# -------------------------
# Telegram Bot State
# -------------------------
# mode per user: 'images' or 'zip'
USER_MODE = {}


def main_menu_keyboard():
    kb = [[InlineKeyboardButton("Images", callback_data='mode_images'),
           InlineKeyboardButton("ZIP", callback_data='mode_zip')]]
    return InlineKeyboardMarkup(kb)


async def send_typing(context: ContextTypes.DEFAULT_TYPE, chat_id: int):
    try:
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    except Exception:
        pass


# -------------------------
# Handlers
# -------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    USER_MODE[user.id] = None
    cleanup_user(user.id)
    await update.message.reply_text(
        "Welcome! Choose what you will upload:",
        reply_markup=main_menu_keyboard()
    )


async def mode_selector(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    if query.data == 'mode_images':
        USER_MODE[user_id] = 'images'
        cleanup_user(user_id)
        _, uploads, _, _ = get_user_dirs(user_id)
        await query.edit_message_text(
            "Mode set to Images. Send images one by one as photo or as image documents. When done, send /process",
        )
    elif query.data == 'mode_zip':
        USER_MODE[user_id] = 'zip'
        cleanup_user(user_id)
        await query.edit_message_text(
            "Mode set to ZIP. Send a .zip file containing images to process.",
        )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    mode = USER_MODE.get(user.id)
    if mode != 'images':
        return
    await send_typing(context, update.effective_chat.id)
    _, uploads, _, _ = get_user_dirs(user.id)
    try:
        photo = update.message.photo[-1]  # highest resolution
        file = await context.bot.get_file(photo.file_id)
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f"image_{ts}.jpg"
        dest_path = os.path.join(uploads, filename)
        await file.download_to_drive(dest_path)
        await update.message.reply_text(f"Saved: {filename}")
    except Exception as e:
        await update.message.reply_text(f"Failed to save image: {e}")


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    doc = update.message.document
    filename = doc.file_name or "file"
    mode = USER_MODE.get(user.id)

    # If user chose ZIP mode, expect a zip
    if mode == 'zip':
        if not allowed_zip_file(filename):
            await update.message.reply_text("Please send a .zip file in ZIP mode.")
            return
        await send_typing(context, update.effective_chat.id)
        base, uploads, translated, archives = get_user_dirs(user.id)
        try:
            file = await context.bot.get_file(doc.file_id)
            zip_path = os.path.join(uploads, secure_filename(filename))
            await file.download_to_drive(zip_path)
            # Extract and process
            safe_extract(zip_path, uploads)
            try:
                os.remove(zip_path)
            except Exception:
                pass
            await process_and_send(update, context, upload_base_name=os.path.splitext(filename)[0])
        except Exception as e:
            await update.message.reply_text(f"Failed to process zip: {e}")
        return

    # If user chose Images mode, allow image documents (png/jpg/etc.)
    if mode == 'images':
        if not allowed_image_file(filename):
            await update.message.reply_text("Send images (png/jpg/webp/bmp/tiff) or use /process when done.")
            return
        await send_typing(context, update.effective_chat.id)
        _, uploads, _, _ = get_user_dirs(user.id)
        try:
            file = await context.bot.get_file(doc.file_id)
            safe_name = secure_filename(filename)
            dest_path = os.path.join(uploads, safe_name)
            await file.download_to_drive(dest_path)
            await update.message.reply_text(f"Saved: {safe_name}")
        except Exception as e:
            await update.message.reply_text(f"Failed to save document: {e}")
        return

    # If no mode chosen, prompt menu
    await update.message.reply_text("Choose what to upload:", reply_markup=main_menu_keyboard())


async def process_and_send(update: Update, context: ContextTypes.DEFAULT_TYPE, upload_base_name: str | None = None):
    user = update.effective_user
    base, uploads, translated, archives = get_user_dirs(user.id)
    chat_id = update.effective_chat.id

    await send_typing(context, chat_id)

    # collect all images in uploads
    image_rel_paths = []
    for root, _, files in os.walk(uploads):
        for fn in files:
            if allowed_image_file(fn):
                abs_path = os.path.join(root, fn)
                rel_path = os.path.relpath(abs_path, uploads)
                image_rel_paths.append(rel_path)

    if not image_rel_paths:
        await update.message.reply_text("No images found to process.")
        return

    # ensure OCR
    initialize_ocr()

    # process
    original_text_structure = defaultdict(lambda: defaultdict(list))
    for rel_path in image_rel_paths:
        original_path = os.path.join(uploads, rel_path)
        translated_rel, original_texts = process_image(original_path, rel_path, TARGET_LANGUAGE, translated)
        chapter = os.path.dirname(translated_rel) or 'Images'
        page_name = os.path.basename(translated_rel)
        if original_texts:
            original_text_structure[chapter][page_name].extend(original_texts)

    # archive
    base_name = upload_base_name or f"images_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    archive_base_no_ext = os.path.join(archives, f"translated_{secure_filename(base_name)}")
    shutil.make_archive(archive_base_no_ext, 'zip', translated)
    zip_file_path = f"{archive_base_no_ext}.zip"

    # text file
    text_file_path = create_text_file(original_text_structure, base_name, archives)

    # send back
    try:
        await context.bot.send_document(chat_id=chat_id, document=open(zip_file_path, 'rb'), filename=os.path.basename(zip_file_path), caption="Translated images")
        await context.bot.send_document(chat_id=chat_id, document=open(text_file_path, 'rb'), filename=os.path.basename(text_file_path), caption="Extracted original text")
    except Exception as e:
        await update.message.reply_text(f"Failed to send results: {e}")

    # cleanup
    cleanup_user(user.id)


async def process_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    mode = USER_MODE.get(update.effective_user.id)
    if mode != 'images':
        await update.message.reply_text("Use /start and select Images mode to batch-send images.")
        return
    await process_and_send(update, context)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Use /start to choose upload type.\n- Images: send multiple photos or image documents, then /process.\n- ZIP: send a .zip with images.")


def build_app():
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not token:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN environment variable with your bot token.")
    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler('start', start))
    app.add_handler(CommandHandler('help', help_command))
    app.add_handler(CommandHandler('process', process_command))

    app.add_handler(CallbackQueryHandler(mode_selector))

    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    return app


if __name__ == '__main__':
    try:
        application = build_app()
        print("Bot is running. Press Ctrl+C to stop.")
        application.run_polling(close_loop=False)
    except Exception as e:
        print(f"Failed to start bot: {e}")