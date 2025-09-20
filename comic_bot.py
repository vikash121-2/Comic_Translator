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
import filetype

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

# --- CONFIGURATION ---
BOT_TOKEN = "6298615623:AAEyldSFqE2HT-2vhITBmZ9lQL23C0fu-Ao"  # <-- IMPORTANT: Replace with your bot token
FONT_PATH = "ComicNeue-Bold.ttf"

# --- CONVERSATION STATES ---
(
    MAIN_MENU,
    JSON_MAKER_CHOICE, CHOOSE_LANGUAGE, WAITING_IMAGES_OCR, WAITING_ZIP_OCR,
    JSON_TRANSLATE_CHOICE,
    WAITING_JSON_TRANSLATE_IMG, WAITING_IMAGES_TRANSLATE,
    WAITING_JSON_TRANSLATE_ZIP, WAITING_ZIP_TRANSLATE,
    JSON_DIVIDE_CHOICE,
    WAITING_JSON_DIVIDE, WAITING_ZIP_DIVIDE
) = range(13)

# --- OCR ENGINE SETUP ---
readers = {}
def get_reader(lang_code):
    global readers
    if lang_code not in readers:
        try:
            import easyocr
            logger.info(f"Initializing EasyOCR for language: {lang_code}...")
            readers[lang_code] = easyocr.Reader([lang_code, 'en'], gpu=torch.cuda.is_available())
            logger.info(f"EasyOCR for {lang_code} Initialized.")
        except Exception as e:
            logger.critical(f"Could not load easyocr model for {lang_code}. Error: {e}")
            return None
    return readers[lang_code]

# --- HELPER FUNCTIONS ---
def cleanup_user_data(context: ContextTypes.DEFAULT_TYPE):
    if 'temp_dir_obj' in context.user_data and hasattr(context.user_data.get('temp_dir_obj'), 'cleanup'):
        context.user_data['temp_dir_obj'].cleanup()
    context.user_data.clear()

def draw_text_in_box(draw: ImageDraw, box: List[int], text: str, font_path: str, max_font_size: int = 60):
    box_width, box_height = box[2] - box[0], box[3] - box[1]
    if not text or box_width <= 0 or box_height <= 0: return
    font_size = max_font_size
    font = ImageFont.truetype(font_path, font_size)
    while font_size > 5:
        avg_char_width = font.getlength("a")
        wrap_width = max(1, int(box_width / avg_char_width * 1.8)) if avg_char_width > 0 else 1
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

# --- MAIN MENU & NAVIGATION ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    keyboard = [
        [InlineKeyboardButton("📝 Json maker", callback_data="main_json_maker")],
        [InlineKeyboardButton("🎨 json To Comic translate", callback_data="main_translate")],
        [InlineKeyboardButton("✂️ json divide", callback_data="main_divide")],
    ]
    message_text = "Welcome! This is a folder-aware comic translator. Please choose an option:"
    if update.message:
        await update.message.reply_text(message_text, reply_markup=InlineKeyboardMarkup(keyboard))
    elif update.callback_query:
        query = update.callback_query
        try: await query.answer()
        except BadRequest: logger.info("Callback query already answered.")
        await query.edit_message_text(message_text, reply_markup=InlineKeyboardMarkup(keyboard))
    return MAIN_MENU

async def back_to_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    cleanup_user_data(context)
    return await start(update, context)

# --- 1. Json Maker (Text Extraction) ---
async def json_maker_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    keyboard = [[InlineKeyboardButton("🖼️ Image(s) Upload", callback_data="jm_image")], [InlineKeyboardButton("🗂️ Zip Upload", callback_data="jm_zip")], [InlineKeyboardButton("« Back", callback_data="main_menu_start")]]
    await query.answer()
    await query.edit_message_text("How would you like to provide source files?", reply_markup=InlineKeyboardMarkup(keyboard))
    return JSON_MAKER_CHOICE

async def json_maker_prompt_language(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    context.user_data['workflow'] = query.data.split('_')[1]
    keyboard = [[InlineKeyboardButton("Japanese", callback_data="lang_ja"), InlineKeyboardButton("Korean", callback_data="lang_ko")], [InlineKeyboardButton("Chinese (Simp)", callback_data="lang_ch_sim"), InlineKeyboardButton("Chinese (Trad)", callback_data="lang_ch_tra")]]
    await query.answer()
    await query.edit_message_text("Please select the source language:", reply_markup=InlineKeyboardMarkup(keyboard))
    return CHOOSE_LANGUAGE

async def json_maker_prompt_files(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    context.user_data['lang_code'] = query.data.split('_', 1)[1]
    workflow = context.user_data.get('workflow')
    if workflow == 'image':
        context.user_data['temp_dir_obj'] = tempfile.TemporaryDirectory()
        context.user_data['image_paths'] = []
        await query.answer()
        await query.edit_message_text("Language selected. Please send your images. Press 'Done' when finished.")
        return WAITING_IMAGES_OCR
    elif workflow == 'zip':
        await query.answer()
        await query.edit_message_text("Language selected. Now, please send your single .zip file.")
        return WAITING_ZIP_OCR

async def collect_images(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        temp_dir_path = context.user_data['temp_dir_obj'].name
        image_paths = context.user_data['image_paths']
    except KeyError:
        await update.message.reply_text("Something went wrong. Please start over.")
        return ConversationHandler.END
    file_to_download, file_name = (None, None)
    if update.message.photo:
        file_to_download = await update.message.photo[-1].get_file()
        file_name = f"{file_to_download.file_id}.jpg"
    elif update.message.document and update.message.document.mime_type.startswith('image/'):
        file_to_download = await update.message.document.get_file()
        file_name = update.message.document.file_name
    else: return WAITING_IMAGES_OCR
    file_path = os.path.join(temp_dir_path, file_name)
    await file_to_download.download_to_drive(file_path)
    image_paths.append(file_path)
    keyboard = [[InlineKeyboardButton("✅ Done Uploading", callback_data="process_images")]]
    await update.message.reply_text(f"Image {len(image_paths)} received. Send another, or press Done.", reply_markup=InlineKeyboardMarkup(keyboard))
    return WAITING_IMAGES_OCR

async def process_collected_images(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("Processing images...")
    image_paths = context.user_data.get('image_paths', [])
    lang_code = context.user_data.get('lang_code', 'en')
    ocr_reader = get_reader(lang_code)
    if not ocr_reader:
        await query.edit_message_text("Error: OCR model could not be loaded.")
        return await back_to_main_menu(update, context)
    all_text_data = []
    for img_path in sorted(image_paths):
        filename = os.path.basename(img_path)
        img_np = np.array(Image.open(img_path).convert("RGB"))
        results = ocr_reader.readtext(img_np, paragraph=True, mag_ratio=1.5, text_threshold=0.4)
        for i, (bbox, text) in enumerate(results):
            text_entry = {"filename": filename, "block_id": i, "bbox": [[int(p[0]), int(p[1])] for p in bbox], "original_text": text, "translated_text": ""}
            all_text_data.append(text_entry)
    json_path = os.path.join(context.user_data['temp_dir_obj'].name, "extracted_text.json")
    with open(json_path, 'w', encoding='utf-8') as f: json.dump(all_text_data, f, ensure_ascii=False, indent=4)
    await context.bot.send_document(chat_id=query.message.chat.id, document=open(json_path, 'rb'), caption=f"Extraction complete.")
    cleanup_user_data(context)
    return await start(update, context)

async def json_maker_process_zip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Zip file received. Processing...")
    lang_code = context.user_data.get('lang_code', 'en')
    ocr_reader = get_reader(lang_code)
    if not ocr_reader:
        await update.message.reply_text("Error: OCR model could not be loaded.")
        return await back_to_main_menu(update, context)
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = Path(temp_dir)
        tg_file = await update.message.document.get_file()
        file_path = input_dir / tg_file.file_name
        await tg_file.download_to_drive(file_path)
        with zipfile.ZipFile(file_path, 'r') as zip_ref: zip_ref.extractall(input_dir)
        os.remove(file_path)
        image_paths = [p for p in input_dir.rglob('*') if p.is_file() and filetype.is_image(p)]
        if not image_paths:
            await update.message.reply_text("No compatible images found in the zip.")
            return await back_to_main_menu(update, context)
        all_text_data = []
        for img_path in sorted(image_paths):
            relative_path = img_path.relative_to(input_dir)
            try:
                img_np = np.array(Image.open(img_path).convert("RGB"))
                results = ocr_reader.readtext(img_np, paragraph=True, mag_ratio=1.5, text_threshold=0.4)
                for i, (bbox, text) in enumerate(results):
                    text_entry = {"filename": str(relative_path).replace('\\', '/'),"block_id": i, "bbox": [[int(p[0]), int(p[1])] for p in bbox],"original_text": text,"translated_text": ""}
                    all_text_data.append(text_entry)
            except Exception as e:
                logger.error(f"Error processing {relative_path}: {e}")
        json_path = input_dir / "extracted_text.json"
        with open(json_path, 'w', encoding='utf-8') as f: json.dump(all_text_data, f, ensure_ascii=False, indent=4)
        await update.message.reply_document(document=open(json_path, 'rb'), caption=f"Extraction complete.")
    cleanup_user_data(context)
    return await start(update, context)

# --- 2. Json To Comic Translate ---
async def json_translate_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    keyboard = [[InlineKeyboardButton("🖼️ Image(s) Upload", callback_data="jt_image")], [InlineKeyboardButton("🗂️ Zip Upload", callback_data="jt_zip")], [InlineKeyboardButton("« Back", callback_data="main_menu_start")]]
    await query.answer()
    await query.edit_message_text("How would you like to apply translations?", reply_markup=InlineKeyboardMarkup(keyboard))
    return JSON_TRANSLATE_CHOICE

async def json_translate_prompt_json_for_img(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("First, please upload the translated JSON file.")
    return WAITING_JSON_TRANSLATE_IMG

async def json_translate_get_json_for_img(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    json_file = await update.message.document.get_file()
    context.user_data['json_data'] = json.loads(await json_file.download_as_bytearray())
    context.user_data['temp_dir_obj'] = tempfile.TemporaryDirectory()
    context.user_data['received_images'] = {}
    await update.message.reply_text("JSON received. Now send the original images. Press 'Done' when finished.")
    return WAITING_IMAGES_TRANSLATE

async def json_translate_collect_images(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        temp_dir_path = context.user_data['temp_dir_obj'].name
        received_images = context.user_data['received_images']
    except KeyError:
        await update.message.reply_text("Something went wrong. Please start over.")
        return ConversationHandler.END
    file_to_download, original_filename = (None, None)
    if update.message.photo:
        file_to_download = await update.message.photo[-1].get_file()
        original_filename = f"photo_{file_to_download.file_id}.jpg"
    elif update.message.document and update.message.document.mime_type.startswith('image/'):
        file_to_download = await update.message.document.get_file()
        original_filename = update.message.document.file_name
    else: return WAITING_IMAGES_TRANSLATE
    file_path = os.path.join(temp_dir_path, original_filename)
    await file_to_download.download_to_drive(file_path)
    received_images[original_filename] = file_path
    keyboard = [[InlineKeyboardButton("✅ Done Uploading", callback_data="jt_process_images")]]
    await update.message.reply_text(f"Image '{original_filename}' received. Send another, or press Done.", reply_markup=InlineKeyboardMarkup(keyboard))
    return WAITING_IMAGES_TRANSLATE

async def json_translate_process_images(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("Applying translations to images...")
    json_data = context.user_data.get('json_data', [])
    received_images = context.user_data.get('received_images', {})
    if not received_images:
        await query.edit_message_text("You didn't send any images! Please start over.")
        return await back_to_main_menu(update, context)
    if not os.path.exists(FONT_PATH):
        await context.bot.send_message(chat_id=query.message.chat.id, text=f"CRITICAL ERROR: Font file '{FONT_PATH}' not found!")
        return await back_to_main_menu(update, context)
    images_processed_count = 0
    translations_by_file = {}
    for entry in json_data:
        fname = entry['filename']
        if fname not in translations_by_file: translations_by_file[fname] = []
        translations_by_file[fname].append(entry)
    for uploaded_filename, image_path in received_images.items():
        matched_translations = translations_by_file.get(uploaded_filename) or translations_by_file.get(Path(uploaded_filename).name)
        if matched_translations:
            img = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            for entry in matched_translations:
                bbox, translated_text = entry['bbox'], entry.get('translated_text', '').strip()
                x_coords = [p[0] for p in bbox]; y_coords = [p[1] for p in bbox]
                simple_box = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                draw.rectangle(simple_box, fill="white", outline="black", width=1)
                if translated_text:
                    draw_text_in_box(draw, simple_box, translated_text, FONT_PATH)
            bio = io.BytesIO()
            bio.name = f"translated_{uploaded_filename}"
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
    json_file = await update.message.document.get_file()
    context.user_data['json_data'] = json.loads(await json_file.download_as_bytearray())
    await update.message.reply_text("JSON file received. Now, please upload the original .zip file.")
    return WAITING_ZIP_TRANSLATE

async def json_translate_process_zip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Zip file received. Applying translations...")
    json_data = context.user_data.get('json_data')
    if not json_data:
        await update.message.reply_text("Error: JSON data was lost.")
        return await back_to_main_menu(update, context)
    if not os.path.exists(FONT_PATH):
        await update.message.reply_text(f"CRITICAL ERROR: Font file '{FONT_PATH}' not found!")
        return await back_to_main_menu(update, context)
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = Path(temp_dir) / "input"; output_dir = Path(temp_dir) / "output"
        input_dir.mkdir(); output_dir.mkdir()
        zip_tg_file = await update.message.document.get_file()
        zip_path = input_dir / "images.zip"
        await zip_tg_file.download_to_drive(zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref: zip_ref.extractall(input_dir)
        translations_by_file = {}
        for entry in json_data:
            fname = entry['filename']
            if fname not in translations_by_file: translations_by_file[fname] = []
            translations_by_file[fname].append(entry)
        for rel_path_str, translations in translations_by_file.items():
            img_path = input_dir / Path(rel_path_str)
            if not img_path.exists(): continue
            img = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            for entry in translations:
                bbox, translated_text = entry['bbox'], entry.get('translated_text', '').strip()
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
        await update.message.reply_text("Error: JSON data was lost.")
        return await back_to_main_menu(update, context)
    with tempfile.TemporaryDirectory() as temp_dir:
        working_dir = Path(temp_dir) / "work"
        working_dir.mkdir()
        zip_tg_file = await update.message.document.get_file()
        zip_path = working_dir / "images.zip"
        await zip_tg_file.download_to_drive(zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref: zip_ref.extractall(working_dir)
        blocks_by_folder = {}
        for entry in json_data:
            p = Path(entry['filename'])
            folder_name = str(p.parent)
            if folder_name not in blocks_by_folder: blocks_by_folder[folder_name] = []
            blocks_by_folder[folder_name].append(entry)
        for folder_rel_path, blocks in blocks_by_folder.items():
            folder_abs_path = working_dir / Path(folder_rel_path)
            if not folder_abs_path.is_dir(): continue
            folder_json_path = folder_abs_path / "folder_text.json"
            with open(folder_json_path, 'w', encoding='utf-8') as f: json.dump(blocks, f, ensure_ascii=False, indent=4)
            images_in_folder = {Path(b['filename']).name for b in blocks}
            for img_name in images_in_folder:
                img_path = folder_abs_path / img_name
                if img_path.exists():
                    img = Image.open(img_path).convert("RGB")
                    draw = ImageDraw.Draw(img)
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

# --- APPLICATION SETUP ---
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
                CallbackQueryHandler(json_maker_prompt_language, pattern="^jm_image$"),
                CallbackQueryHandler(json_maker_prompt_language, pattern="^jm_zip$"),
                CallbackQueryHandler(back_to_main_menu, pattern="^main_menu_start$"),
            ],
            CHOOSE_LANGUAGE: [
                CallbackQueryHandler(json_maker_prompt_files, pattern="^lang_"),
            ],
            WAITING_IMAGES_OCR: [
                MessageHandler(filters.PHOTO | filters.Document.IMAGE, collect_images),
                CallbackQueryHandler(process_collected_images, pattern="^process_images$"),
            ],
            WAITING_ZIP_OCR: [MessageHandler(filters.Document.ZIP, json_maker_process_zip)],
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
            JSON_DIVIDE_CHOICE: [
                CallbackQueryHandler(json_divide_prompt_json, pattern="^jd_zip$"),
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
