import logging
import os
import io
import zipfile
import shutil
import json
import textwrap
import torch
import tempfile
from typing import List, Dict
from PIL import Image, ImageDraw, ImageFont, ImageFile
from pathlib import Path
import numpy as np

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import (
    Application,
    CommandHandler,
    ConversationHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)
from telegram.error import BadRequest

ImageFile.LOAD_TRUNCATED_IMAGES = True
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# In Colab, you would use:
# from google.colab import userdata
# BOT_TOKEN = userdata.get('BOT_TOKEN')

BOT_TOKEN = "6298615623:AAEyldSFqE2HT-2vhITBmZ9lQL23C0fu-Ao"  # <-- IMPORTANT: Replace with your bot token
FONT_PATH = "ComicNeue-Bold.ttf"

# Conversation states
(
    MAIN_MENU,
    JSON_MAKER_CHOICE, WAITING_FILES_OCR,
    JSON_TRANSLATE_CHOICE, WAITING_FILES_TRANSLATE,
) = range(5)

# --- Global variables to manage the OCR reader efficiently ---
current_reader_langs = None
reader = None

def get_reader(langs):
    """Initializes or retrieves the EasyOCR reader to avoid reloading."""
    global current_reader_langs, reader
    sorted_langs = tuple(sorted(langs))
    if sorted_langs != current_reader_langs:
        try:
            import easyocr
            logger.info(f"Initializing EasyOCR for languages: {langs}...")
            reader = easyocr.Reader(langs, gpu=torch.cuda.is_available())
            current_reader_langs = sorted_langs
            logger.info("EasyOCR Initialized.")
        except Exception as e:
            logger.critical(f"Could not load easyocr model. Error: {e}")
            return None
    return reader

def cleanup_user_data(context: ContextTypes.DEFAULT_TYPE):
    """Cleans up temporary data at the end of a conversation."""
    for key in ['temp_dir_obj', 'json_data', 'files_to_process', 'lang_code']:
        if key in context.user_data:
            if hasattr(context.user_data.get(key), 'cleanup'):
                context.user_data[key].cleanup()
            del context.user_data[key]

# --- Main Menu & Core Navigation ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    keyboard = [
        [InlineKeyboardButton("📝 Step 1: Extract Text", callback_data="main_json_maker")],
        [InlineKeyboardButton("🎨 Step 2: Apply Translations", callback_data="main_translate")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    message_text = "Welcome! This is a folder-aware comic translator. Please choose an option:"
    if update.message:
        await update.message.reply_text(message_text, reply_markup=reply_markup)
    elif update.callback_query:
        query = update.callback_query
        try: await query.answer()
        except BadRequest: logger.info("Callback query already answered.")
        await query.edit_message_text(message_text, reply_markup=reply_markup)
    return MAIN_MENU

async def back_to_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    cleanup_user_data(context)
    return await start(update, context)

# --- 1. Json Maker (Text Extraction) ---

async def json_maker_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    keyboard = [
        [
            InlineKeyboardButton("English", callback_data="lang_en"),
            InlineKeyboardButton("Japanese", callback_data="lang_ja"),
            InlineKeyboardButton("Korean", callback_data="lang_ko"),
        ],
        [
            InlineKeyboardButton("Chinese (Simp)", callback_data="lang_ch_sim"),
            InlineKeyboardButton("Chinese (Trad)", callback_data="lang_ch_tra"),
        ],
        [InlineKeyboardButton("« Back", callback_data="main_menu_start")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.answer()
    await query.edit_message_text("Please select the source language of your comic:", reply_markup=reply_markup)
    return JSON_MAKER_CHOICE

async def json_maker_prompt_files(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    context.user_data['lang_code'] = query.data.split('_', 1)[1]
    await query.answer()
    await query.edit_message_text("Language selected. Now, please send your images or a single .zip file.")
    return WAITING_FILES_OCR

async def extract_text_from_files(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Processes uploaded files (images or zip), extracts text, and returns a JSON."""
    await update.message.reply_text("Files received. Processing...")
    
    lang_code = context.user_data.get('lang_code', 'en')
    ocr_reader = get_reader([lang_code, 'en'])
    if ocr_reader is None:
        await update.message.reply_text("Error: OCR model could not be loaded. Please check the logs.")
        return await back_to_main_menu(update, context)

    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = Path(temp_dir)
        
        file_to_process = update.message.document or update.message.photo
        if file_to_process:
            if isinstance(file_to_process, list):
                file_to_process = file_to_process[-1]

            tg_file = await file_to_process.get_file()
            file_name = getattr(file_to_process, 'file_name', f"{tg_file.file_id}.jpg")
            file_path = input_dir / file_name
            await tg_file.download_to_drive(file_path)

            if file_path.suffix.lower() == '.zip':
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(input_dir)
                os.remove(file_path)

        image_paths = list(input_dir.rglob('*.jpg')) + list(input_dir.rglob('*.jpeg')) + list(input_dir.rglob('*.png'))
        if not image_paths:
            await update.message.reply_text("No images found to process.")
            return await back_to_main_menu(update, context)
            
        all_text_data = []
        for img_path in sorted(image_paths):
            relative_path = img_path.relative_to(input_dir)
            logger.info(f"Extracting text from: {relative_path}")
            try:
                img_np = np.array(Image.open(img_path).convert("RGB"))
                results = ocr_reader.readtext(img_np, paragraph=True, mag_ratio=1.5, text_threshold=0.4)

                for i, (bbox, text, prob) in enumerate(results):
                    text_entry = {
                        "filename": str(relative_path).replace('\\', '/'),
                        "block_id": i,
                        "bbox": [[int(p[0]), int(p[1])] for p in bbox],
                        "original_text": text,
                        "translated_text": ""
                    }
                    all_text_data.append(text_entry)
            except Exception as e:
                logger.error(f"Error processing {relative_path}: {e}")

        json_path = input_dir / "extracted_text.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_text_data, f, ensure_ascii=False, indent=4)

        await update.message.reply_document(document=open(json_path, 'rb'), caption=f"Extraction complete. Found text in {len(image_paths)} images.")

    cleanup_user_data(context)
    return await start(update, context)

# --- 2. Json To Comic Translate Feature ---

async def json_translate_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("Please send the **original images/zip** and the **translated JSON file** together in one message.")
    return WAITING_FILES_TRANSLATE

async def apply_translations_to_images(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not update.message.media_group_id:
        await update.message.reply_text("Please send the files together. Select the images/zip and the JSON, then send.")
        return WAITING_FILES_TRANSLATE

    context.user_data.setdefault('files_to_process', {})[update.message.media_group_id] = context.user_data.get(update.message.media_group_id, [])
    
    file_to_process = update.message.document or (update.message.photo[-1] if update.message.photo else None)
    if file_to_process:
        context.user_data['files_to_process'][update.message.media_group_id].append(file_to_process)

    job_name = f"process_translation_{update.message.media_group_id}"
    if not context.job_queue.get_jobs_by_name(job_name):
        context.job_queue.run_once(process_translation_job, 2, data={'media_group_id': update.message.media_group_id, 'chat_id': update.effective_chat.id}, name=job_name)

    return WAITING_FILES_TRANSLATE

async def process_translation_job(context: ContextTypes.DEFAULT_TYPE):
    job_data = context.job.data
    media_group_id = job_data['media_group_id']
    chat_id = job_data['chat_id']
    
    files = context.user_data.get('files_to_process', {}).get(media_group_id, [])
    if not files:
        await context.bot.send_message(chat_id, "Something went wrong collecting files.")
        return

    await context.bot.send_message(chat_id, "Files received. Applying translations...")

    json_file, image_files = None, []
    for file in files:
        if hasattr(file, 'file_name') and file.file_name and file.file_name.lower().endswith('.json'):
            json_file = file
        else:
            image_files.append(file)
    
    if not json_file or not image_files:
        await context.bot.send_message(chat_id, "Error: You must provide at least one image/zip and one JSON file.")
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir, output_dir = Path(temp_
