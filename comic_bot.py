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
    JSON_DIVIDE_CHOICE, 
    WAITING_JSON_DIVIDE, WAITING_ZIP_DIVIDE
) = range(9)

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
    for key in ['temp_dir_obj', 'json_data', 'lang_code', 'zip_file']:
        if key in context.user_data:
            if hasattr(context.user_data.get(key), 'cleanup'):
                context.user_data[key].cleanup()
            del context.user_data[key]

def draw_text_in_box(draw: ImageDraw, box: List[int], text: str, font_path: str, max_font_size: int = 60):
    box_width, box_height = box[2] - box[0], box[3] - box[1]
    if not text or box_width <= 0 or box_height <= 0: return
    font_size = max_font_size
    font = ImageFont.truetype(font_path, font_size)
    while font_size > 5:
        avg_char_width = font.getlength("a")
        wrap_width = max(1, int(box_width / avg_char_width)) if avg_char_width > 0 else 1
        wrapped_text = textwrap.wrap(text, width=wrap_width, break_long_words=True)
        line_heights = [draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1] if line else font_size for line in wrapped_text]
        total_text_height = sum(line_heights)
        if total_text_height <= box_height and all(font.getlength(line) <= box_width for line in wrapped_text):
            break
        font_size -= 2
        font = ImageFont.truetype(font_path, font_size)
    y_start = box[1] + (box_height - total_text_height) / 2
    for i, line in enumerate(wrapped_text):
        line_width = font.getlength(line)
        x_start = box[0] + (box_width - line_width) / 2
        draw.text((x_start, y_start), line, font=font, fill="black", anchor="lt")
        y_start += line_heights[i]

# --- Main Menu & Core Navigation ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    keyboard = [
        [InlineKeyboardButton("📝 Json maker", callback_data="main_json_maker")],
        [InlineKeyboardButton("🎨 json To Comic translate", callback_data="main_translate")],
        [InlineKeyboardButton("✂️ json divide", callback_data="main_divide")],
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
        [InlineKeyboardButton("Chinese (Simp)", callback_data="lang_ch_sim"), InlineKeyboardButton("Chinese (Trad)", callback_data="lang_ch_tra")],
        [InlineKeyboardButton("Japanese", callback_data="lang_ja"), InlineKeyboardButton("Korean", callback_data="lang_ko")],
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
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("Please upload the translated JSON file.")
    return WAITING_JSON_TRANSLATE_ZIP

async def json_translate_get_json(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("JSON file received. Now, please upload the original .zip file with the images.")
    json_file = await update.message.document.get_file()
    context.user_data['json_data'] = json.loads(await json_file.download_as_bytearray())
    return WAITING_ZIP_TRANSLATE

async def json_translate_process_zip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Zip file received. Applying translations...")
    json_data = context.user_data.get('json_data')
    if not json_data:
        await update.message.reply_text("Error: JSON data was lost. Please start over.")
        return await back_to_main_menu(update, context)
        
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = Path(temp_dir) / "input"
        output_dir = Path(temp_dir) / "output"
        input_dir.mkdir(); output_dir.mkdir()
        zip_tg_file = await update.message.document.get_file()
        zip_path = input_dir / "images.zip"
        await zip_tg_file.download_to_drive(zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(input_dir)

        translations_by_file = {}
        for entry in json_data:
            fname = entry['filename']
            if fname not in translations_by_file: translations_by_file[fname] = []
            translations_by_file[fname].append(entry)
        
        try:
            font = ImageFont.truetype(FONT_PATH, 40)
        except IOError:
            await update.message.reply_text(f"Error: Font '{FONT_PATH}' not found.")
            return await back_to_main_menu(update, context)

        for rel_path_str, translations in translations_by_file.items():
            img_path = input_dir / Path(rel_path_str)
            if not img_path.exists():
                logger.warning(f"Image '{rel_path_str}' from JSON not found.")
                continue
            img = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            for entry in translations:
                bbox = entry['bbox']
                translated_text = entry.get('translated_text', '').strip()
                x_coords = [p[0] for p in bbox]; y_coords = [p[1] for p in bbox]
                simple_box = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                draw.rectangle(simple_box, fill="white", outline="black", width=1)
                if translated_text:
                    draw_text_in_box(draw, simple_box, translated_text, FONT_PATH)
            output_path = output_dir / Path(rel_path_str)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path)
        zip_path_str = os.path.join(temp_dir, "final_translated_comics")
        shutil.make_archive(zip_path_str, 'zip', output_dir)
        await update.message.reply_document(document=open(f"{zip_path_str}.zip", 'rb'), caption="Processing complete!")
    cleanup_user_data(context)
    return await start(update, context)


# --- 3. Json Divide Feature ---

async def json_divide_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("This feature requires a master JSON and a zip file.\nPlease upload the master JSON file first.")
    return WAITING_JSON_DIVIDE

async def json_divide_get_json(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("JSON file received. Now, please upload the original zip file.")
    json_file = await update.message.document.get_file()
    context.user_data['json_data'] = json.loads(await json_file.download_as_bytearray())
    return WAITING_ZIP_DIVIDE

async def json_divide_process_zip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Zip file received. Dividing JSON and masking images...")
    json_data = context.user_data.get('json_data')
    if not json_data:
        await update.message.reply_text("Error: JSON data was lost. Please start over.")
        return await back_to_main_menu(update, context)

    with tempfile.TemporaryDirectory() as temp_dir:
        working_dir = Path(temp_dir) / "work"
        working_dir.mkdir()
        
        zip_tg_file = await update.message.document.get_file()
        zip_path = working_dir / "images.zip"
        await zip_tg_file.download_to_drive(zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(working_dir)

        # Group text blocks by their parent folder
        blocks_by_folder = {}
        for entry in json_data:
            p = Path(entry['filename'])
            folder_name = str(p.parent)
            if folder_name not in blocks_by_folder: blocks_by_folder[folder_name] = []
            blocks_by_folder[folder_name].append(entry)

        for folder_rel_path, blocks in blocks_by_folder.items():
            folder_abs_path = working_dir / Path(folder_rel_path)
            if not folder_abs_path.is_dir(): continue
            
            # 1. Create and save the smaller JSON
            folder_json_path = folder_abs_path / "folder_text.json"
            with open(folder_json_path, 'w', encoding='utf-8') as f:
                json.dump(blocks, f, ensure_ascii=False, indent=4)
            
            # 2. Mask the images in this folder
            images_in_folder = {Path(b['filename']).name for b in blocks}
            for img_name in images_in_folder:
                img_path = folder_abs_path / img_name
                if img_path.exists():
                    img = Image.open(img_path).convert("RGB")
                    draw = ImageDraw.Draw(img)
                    # Get all boxes for this specific image
                    boxes_to_mask = [b['bbox'] for b in blocks if Path(b['filename']).name == img_name]
                    for bbox_points in boxes_to_mask:
                        x_coords = [p[0] for p in bbox_points]; y_coords = [p[1] for p in bbox_points]
                        simple_box = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                        draw.rectangle(simple_box, fill="black")
                    img.save(img_path)
        
        zip_path_str = os.path.join(temp_dir, "final_divided_comics")
        shutil.make_archive(zip_path_str, 'zip', working_dir)
        await update.message.reply_document(document=open(f"{zip_path_str}.zip", 'rb'), caption="Dividing complete!")

    cleanup_user_data(context)
    return await start(update, context)


# --- Main Application Setup ---

def main() -> None:
    application = Application.builder().token(BOT_TOKEN).build()
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            MAIN_MENU: [
                CallbackQueryHandler(json_maker_menu, pattern="^main_json_maker$"),
                CallbackQueryHandler(json_translate_menu, pattern="^main_translate$"),
                CallbackQueryHandler(json_divide_menu, pattern="^main_divide$"),
                CallbackQueryHandler(back_to_main_menu, pattern="^main_menu_start$"), 
            ],
            JSON_MAKER_CHOICE: [
                CallbackQueryHandler(json_maker_prompt_files, pattern="^lang_"),
                CallbackQueryHandler(back_to_main_menu, pattern="^main_menu_start$"),
            ],
            WAITING_FILES_OCR: [MessageHandler(filters.PHOTO | filters.Document.ALL, extract_text_from_files)],
            
            JSON_TRANSLATE_CHOICE: [
                CallbackQueryHandler(json_translate_menu, pattern="^main_translate$") # Re-entry point
            ],
            WAITING_JSON_TRANSLATE_ZIP: [MessageHandler(filters.Document.FileExtension("json"), json_translate_get_json)],
            WAITING_ZIP_TRANSLATE: [MessageHandler(filters.Document.ZIP, apply_translations_to_zip)],

            JSON_DIVIDE_CHOICE: [
                CallbackQueryHandler(json_divide_prompt_json, pattern="^jd_zip$"), # Placeholder, should be integrated if needed
                CallbackQueryHandler(back_to_main_menu, pattern="^main_menu_start$")
            ],
            WAITING_JSON_DIVIDE: [MessageHandler(filters.Document.FileExtension("json"), json_divide_get_json)],
            WAITING_ZIP_DIVIDE: [MessageHandler(filters.Document.ZIP, json_divide_process_zip)],
        },
        fallbacks=[CommandHandler("start", start)],
    )
    application.add_handler(conv_handler)
    application.run_polling()

if __name__ == "__main__":
    main()
