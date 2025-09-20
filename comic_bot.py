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

# All conversation states are now defined
(
    MAIN_MENU,
    JSON_MAKER_CHOICE, WAITING_FILES_OCR,
    JSON_TRANSLATE_CHOICE, 
    WAITING_JSON_TRANSLATE_ZIP, WAITING_ZIP_TRANSLATE,
    WAITING_JSON_TRANSLATE_IMG, WAITING_IMAGES_TRANSLATE
) = range(8)

# --- Global variables for OCR ---
current_reader_langs = None
reader = None

def get_reader(langs):
    """Initializes or retrieves the EasyOCR reader."""
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
    """Cleans up temporary data."""
    for key in ['temp_dir_obj', 'json_data', 'lang_code', 'received_images']:
        if key in context.user_data:
            if hasattr(context.user_data.get(key), 'cleanup'):
                context.user_data[key].cleanup()
            del context.user_data[key]

def draw_text_in_box(draw: ImageDraw, box: List[int], text: str, font_path: str, max_font_size: int = 60):
    # ... (This function is complete and correct from the previous version)
    pass

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
        [InlineKeyboardButton("Japanese", callback_data="lang_ja"), InlineKeyboardButton("Korean", callback_data="lang_ko")],
        [InlineKeyboardButton("Chinese (Simp)", callback_data="lang_ch_sim"), InlineKeyboardButton("Chinese (Trad)", callback_data="lang_ch_tra")],
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
    await update.message.reply_text("Files received. Processing...")
    lang_code = context.user_data.get('lang_code', 'en')
    ocr_reader = get_reader([lang_code, 'en'])
    if ocr_reader is None:
        await update.message.reply_text("Error: OCR model could not be loaded.")
        return await back_to_main_menu(update, context)

    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = Path(temp_dir)
        file_to_process = update.message.document or update.message.photo
        if file_to_process:
            if isinstance(file_to_process, list): file_to_process = file_to_process[-1]
            tg_file = await file_to_process.get_file()
            file_name = getattr(file_to_process, 'file_name', f"{tg_file.file_id}.jpg")
            file_path = input_dir / file_name
            await tg_file.download_to_drive(file_path)
            if file_path.suffix.lower() == '.zip':
                with zipfile.ZipFile(file_path, 'r') as zip_ref: zip_ref.extractall(input_dir)
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
                for i, (bbox, text) in enumerate(results):
                    text_entry = {"filename": str(relative_path).replace('\\', '/'),"block_id": i, "bbox": [[int(p[0]), int(p[1])] for p in bbox],"original_text": text,"translated_text": ""}
                    all_text_data.append(text_entry)
            except Exception as e:
                logger.error(f"Error processing {relative_path}: {e}")
        json_path = input_dir / "extracted_text.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_text_data, f, ensure_ascii=False, indent=4)
        await update.message.reply_document(document=open(json_path, 'rb'), caption=f"Extraction complete.")
    cleanup_user_data(context)
    return await start(update, context)

# --- 2. Json To Comic Translate Feature ---

async def json_translate_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """RESTORED: Shows both Image and Zip upload options."""
    query = update.callback_query
    keyboard = [
        [
            InlineKeyboardButton("🖼️ Image Upload", callback_data="jt_image"),
            InlineKeyboardButton("🗂️ Zip Upload", callback_data="jt_zip")
        ],
        [InlineKeyboardButton("« Back", callback_data="main_menu_start")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.answer()
    await query.edit_message_text("How would you like to apply translations?", reply_markup=reply_markup)
    return JSON_TRANSLATE_CHOICE

async def json_translate_prompt_json_for_img(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Asks for JSON file before image upload."""
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("Please upload the translated JSON file first.")
    return WAITING_JSON_TRANSLATE_IMG

async def json_translate_get_json_for_img(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Receives JSON, then asks for images."""
    json_file = await update.message.document.get_file()
    context.user_data['json_data'] = json.loads(await json_file.download_as_bytearray())
    context.user_data['temp_dir_obj'] = tempfile.TemporaryDirectory()
    context.user_data['received_images'] = {}
    await update.message.reply_text("JSON received. Now send the original images. Press 'Done' when finished.")
    return WAITING_IMAGES_TRANSLATE

async def json_translate_collect_images(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Collects images one by one for translation."""
    try:
        temp_dir_path = context.user_data['temp_dir_obj'].name
        received_images = context.user_data['received_images']
    except KeyError:
        await update.message.reply_text("Something went wrong. Please start over.")
        return ConversationHandler.END
    
    file_to_download, original_filename = (None, None)
    if update.message.photo:
        file_to_download = await update.message.photo[-1].get_file()
        original_filename = f"{file_to_download.file_id}.jpg" # Use a unique name for photos
    elif update.message.document and update.message.document.mime_type.startswith('image/'):
        file_to_download = await update.message.document.get_file()
        original_filename = update.message.document.file_name
    else: return WAITING_IMAGES_TRANSLATE

    file_path = os.path.join(temp_dir_path, original_filename)
    await file_to_download.download_to_drive(file_path)
    # The JSON refers to filenames. We need to match based on that.
    # For photos, this is tricky. We'll rely on the order or a single photo.
    # For now, we store it under its original name (or a generated one).
    received_images[original_filename] = file_path
    
    keyboard = [[InlineKeyboardButton("✅ Done Uploading", callback_data="jt_process_images")]]
    await update.message.reply_text(f"Image '{original_filename}' received. Send another, or press Done.", reply_markup=InlineKeyboardMarkup(keyboard))
    return WAITING_IMAGES_TRANSLATE

async def json_translate_process_images(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Processes collected images for translation."""
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("Applying translations to images...")
    
    json_data = context.user_data.get('json_data', [])
    received_images = context.user_data.get('received_images', {})
    if not received_images:
        await query.edit_message_text("You didn't send any images! Please start over.")
        cleanup_user_data(context)
        return await start(update, context)

    try: font = ImageFont.truetype(FONT_PATH, 40)
    except IOError:
        await context.bot.send_message(chat_id=query.message.chat.id, text=f"Error: Font '{FONT_PATH}' not found.")
        return await back_to_main_menu(update, context)

    images_processed_count = 0
    # Group translations by filename from the flat JSON list
    translations_by_file = {}
    for entry in json_data:
        fname = entry['filename']
        if fname not in translations_by_file: translations_by_file[fname] = []
        translations_by_file[fname].append(entry)

    for filename, image_path in received_images.items():
        if filename in translations_by_file:
            img = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            for entry in translations_by_file[filename]:
                bbox = entry['bbox']
                translated_text = entry.get('translated_text', '').strip()
                top_left, bottom_right = tuple(bbox[0]), tuple(bbox[2])
                draw.rectangle([top_left, bottom_right], fill="white", outline="black", width=1)
                if translated_text:
                    draw_text_in_box(draw, [top_left[0], top_left[1], bottom_right[0], bottom_right[1]], translated_text, FONT_PATH)
            
            bio = io.BytesIO()
            bio.name = f"translated_{filename}"
            img.save(bio, 'JPEG')
            bio.seek(0)
            await context.bot.send_document(chat_id=query.message.chat.id, document=bio)
            images_processed_count += 1

    if images_processed_count == 0:
        await context.bot.send_message(chat_id=query.message.chat.id, text="Warning: No matching filenames found between your JSON and uploaded images.")
    else:
        await context.bot.send_message(chat_id=query.message.chat.id, text="Translation complete!")
    
    cleanup_user_data(context)
    return await start(update, context)


async def json_translate_prompt_json_for_zip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("Please upload the translated JSON file.")
    return WAITING_JSON_TRANSLATE_ZIP

async def json_translate_get_json_for_zip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("JSON file received. Now, please upload the original .zip file with the images.")
    json_file = await update.message.document.get_file()
    context.user_data['json_data'] = json.loads(await json_file.download_as_bytearray())
    return WAITING_ZIP_TRANSLATE

async def json_translate_process_zip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # This is now the handler for the zip file after the JSON has been received.
    await update.message.reply_text("Zip file received. Applying translations...")
    # ... (The rest of the zip processing logic from the Gradio script version)
    pass


def main() -> None:
    application = Application.builder().token(BOT_TOKEN).build()
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            MAIN_MENU: [
                CallbackQueryHandler(json_maker_menu, pattern="^main_json_maker$"),
                CallbackQueryHandler(json_translate_menu, pattern="^main_translate$"),
                # Add json_divide_menu handler here
                CallbackQueryHandler(back_to_main_menu, pattern="^main_menu_start$"), 
            ],
            JSON_MAKER_CHOICE: [
                CallbackQueryHandler(json_maker_prompt_files, pattern="^lang_"),
                CallbackQueryHandler(back_to_main_menu, pattern="^main_menu_start$"),
            ],
            WAITING_FILES_OCR: [MessageHandler(filters.PHOTO | filters.Document.ALL, extract_text_from_files)],
            
            # States for Translation Feature
            JSON_TRANSLATE_CHOICE: [
                CallbackQueryHandler(json_translate_prompt_json_for_img, pattern="^jt_image$"),
                CallbackQueryHandler(json_translate_prompt_json_for_zip, pattern="^jt_zip$"),
                CallbackQueryHandler(back_to_main_menu, pattern="^main_menu_start$"),
            ],
            WAITING_JSON_TRANSLATE_IMG: [MessageHandler(filters.Document.FileExtension("json"), json_translate_get_json_for_img)],
            WAITING_IMAGES_TRANSLATE: [
                MessageHandler(filters.PHOTO | filters.Document.IMAGE, json_translate_collect_images),
                CallbackQueryHandler(json_translate_process_images, pattern="^jt_process_images$"),
            ],
            WAITING_JSON_TRANSLATE_ZIP: [MessageHandler(filters.Document.FileExtension("json"), json_translate_get_json_for_zip)],
            WAITING_ZIP_TRANSLATE: [MessageHandler(filters.Document.ZIP, json_translate_process_zip)],
            
            # States for Divide Feature (can be added here)
        },
        fallbacks=[CommandHandler("start", start)],
    )
    application.add_handler(conv_handler)
    application.run_polling()

if __name__ == "__main__":
    # Note: This is a partial script focusing on the requested fix.
    # The full bodies of unchanged functions have been omitted for clarity.
    # You must merge this with your last complete script.
    pass
