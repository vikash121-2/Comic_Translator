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
import cv2
import asyncio

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
    JSON_MAKER_CHOICE, WAITING_FILES_OCR,
    JSON_TRANSLATE_CHOICE, 
    WAITING_JSON_TRANSLATE_ZIP, WAITING_ZIP_TRANSLATE,
    JSON_DIVIDE_CHOICE, 
    WAITING_JSON_DIVIDE, WAITING_ZIP_DIVIDE
) = range(9)

# --- OCR ENGINE SETUP ---
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

# --- HELPER FUNCTIONS ---
def cleanup_user_data(context: ContextTypes.DEFAULT_TYPE):
    for key in ['temp_dir_obj', 'json_data', 'lang_code', 'zip_file']:
        if key in context.user_data and hasattr(context.user_data.get(key), 'cleanup'):
            context.user_data[key].cleanup()
    context.user_data.clear()

async def send_progress_update(message, processed_count, total_count, feature_name):
    """Sends a progress update message, avoiding rate limits."""
    try:
        progress_text = f"[{feature_name}] Progress: {processed_count} / {total_count} images processed."
        await message.edit_text(progress_text)
        await asyncio.sleep(0.1) # Be nice to Telegram's API
    except Exception as e:
        logger.warning(f"Could not update progress message: {e}")

def draw_text_in_box(draw: ImageDraw, box: List[int], text: str, font_path: str, max_font_size: int = 60):
    """Draws wrapped, centered, auto-sized text with a white border."""
    box_width, box_height = box[2] - box[0], box[3] - box[1]
    if not text or box_width <= 10 or box_height <= 10: return
    font_size = max_font_size
    
    while font_size > 5:
        font = ImageFont.truetype(font_path, font_size)
        avg_char_width = font.getlength("a")
        wrap_width = max(1, int(box_width / avg_char_width * 1.8)) # Adjust multiplier for better wrapping
        wrapped_text = textwrap.wrap(text, width=wrap_width, break_long_words=True)
        
        line_heights = []
        max_line_width = 0
        for line in wrapped_text:
            line_bbox = draw.textbbox((0, 0), line, font=font)
            line_width = line_bbox[2] - line_bbox[0]
            line_heights.append(line_bbox[3] - line_bbox[1] + int(font_size * 0.2)) # Add spacing
            if line_width > max_line_width: max_line_width = line_width
        total_text_height = sum(line_heights)

        if total_text_height <= box_height and max_line_width <= box_width:
            break
        font_size -= 2
    
    y_start = box[1] + (box_height - total_text_height) / 2
    for i, line in enumerate(wrapped_text):
        line_bbox = draw.textbbox((0,0), line, font=font)
        line_width = line_bbox[2] - line_bbox[0]
        x_start = box[0] + (box_width - line_width) / 2
        
        # Draw white border by drawing text at slightly offset positions
        border_color = "white"
        draw.text((x_start-1, y_start-1), line, font=font, fill=border_color, anchor="lt")
        draw.text((x_start+1, y_start-1), line, font=font, fill=border_color, anchor="lt")
        draw.text((x_start-1, y_start+1), line, font=font, fill=border_color, anchor="lt")
        draw.text((x_start+1, y_start+1), line, font=font, fill=border_color, anchor="lt")
        
        # Draw the main black text
        draw.text((x_start, y_start), line, font=font, fill="black", anchor="lt")
        y_start += line_heights[i]

# --- MAIN MENU & NAVIGATION ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    keyboard = [
        [InlineKeyboardButton("📝 Json maker", callback_data="main_json_maker")],
        [InlineKeyboardButton("🎨 json To Comic translate", callback_data="main_translate")],
        [InlineKeyboardButton("✂️ json divide", callback_data="main_divide")],
    ]
    await (update.message or update.callback_query.message).reply_text("Welcome! Please choose an option:", reply_markup=InlineKeyboardMarkup(keyboard))
    if update.callback_query: await update.callback_query.answer()
    return MAIN_MENU

async def back_to_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    cleanup_user_data(context)
    # Edit the message to show the main menu
    query = update.callback_query
    await query.answer()
    keyboard = [[InlineKeyboardButton("📝 Json maker", callback_data="main_json_maker")],[InlineKeyboardButton("🎨 json To Comic translate", callback_data="main_translate")],[InlineKeyboardButton("✂️ json divide", callback_data="main_divide")]]
    await query.edit_message_text("Welcome! Please choose an option:", reply_markup=InlineKeyboardMarkup(keyboard))
    return MAIN_MENU

# --- FEATURE 1: JSON MAKER ---
async def json_maker_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    keyboard = [
        [InlineKeyboardButton("Chinese (Simp)", callback_data="lang_ch_sim"), InlineKeyboardButton("Chinese (Trad)", callback_data="lang_ch_tra")],
        [InlineKeyboardButton("Japanese", callback_data="lang_ja"), InlineKeyboardButton("Korean", callback_data="lang_ko")],
        [InlineKeyboardButton("« Back", callback_data="main_menu_start")]
    ]
    await query.answer()
    await query.edit_message_text("Please select the source language:", reply_markup=InlineKeyboardMarkup(keyboard))
    return JSON_MAKER_CHOICE

async def json_maker_prompt_files(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    context.user_data['lang_code'] = query.data.split('_', 1)[1]
    await query.answer()
    await query.edit_message_text("Language selected. Please send a single .zip file.")
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
            if isinstance(file_to_process, list):
                file_to_process = file_to_process[-1]

            # --- CORRECTED LOGIC ---
            # Get the filename from the original object (document or photo) first.
            # For photos, which don't have a filename, create a unique one using its file_id.
            file_name = getattr(file_to_process, 'file_name', f"{file_to_process.file_id}.jpg")
            
            # Now, get the file object itself to download it
            tg_file = await file_to_process.get_file()
            # --- END OF FIX ---

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
    
# --- FEATURE 2: JSON TO COMIC TRANSLATE ---
async def json_translate_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("This feature applies translated text to a zip of images.\nPlease upload your translated JSON file first.")
    return WAITING_JSON_TRANSLATE_ZIP

async def json_translate_get_json(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("JSON file received. Now, please upload the original .zip file with the images.")
    json_file = await update.message.document.get_file()
    context.user_data['json_data'] = json.loads(await json_file.download_as_bytearray())
    return WAITING_ZIP_TRANSLATE

async def json_translate_process_zip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    progress_message = await update.message.reply_text("Zip file received. Applying translations...")
    json_data = context.user_data.get('json_data')
    if not json_data:
        await progress_message.edit_text("Error: JSON data was lost. Please start over.")
        return await back_to_main_menu(update, context)
        
    if not os.path.exists(FONT_PATH):
        await progress_message.edit_text(f"CRITICAL ERROR: Font file '{FONT_PATH}' not found!")
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

        all_image_paths = [p for p in input_dir.rglob('*') if filetype.is_image(p)]
        total_images = len(all_image_paths)
        processed_count = 0

        for rel_path_str, translations in translations_by_file.items():
            img_path = input_dir / Path(rel_path_str)
            if not img_path.exists(): continue

            img_cv = cv2.imread(str(img_path))
            # Create a mask for inpainting
            mask = np.zeros(img_cv.shape[:2], dtype=np.uint8)
            for entry in translations:
                bbox = entry['bbox']
                points = np.array(bbox, dtype=np.int32)
                cv2.fillPoly(mask, [points], 255)
            
            # Inpaint the original image to remove text
            inpainted_img_cv = cv2.inpaint(img_cv, mask, 3, cv2.INPAINT_TELEA)
            
            # Convert to PIL for text drawing
            img = Image.fromarray(cv2.cvtColor(inpainted_img_cv, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img)

            for entry in translations:
                bbox = entry['bbox']
                translated_text = entry.get('translated_text', '').strip()
                if translated_text:
                    x_coords = [p[0] for p in bbox]; y_coords = [p[1] for p in bbox]
                    simple_box = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                    draw_text_in_box(draw, simple_box, translated_text, FONT_PATH)
            
            output_path = output_dir / Path(rel_path_str)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path)
            processed_count +=1
            if processed_count % 5 == 0:
                await send_progress_update(progress_message, processed_count, total_images, "Translation")

        zip_path_str = os.path.join(temp_dir, "final_translated_comics")
        shutil.make_archive(zip_path_str, 'zip', output_dir)
        await progress_message.delete()
        await update.message.reply_document(document=open(f"{zip_path_str}.zip", 'rb'), caption="Processing complete!")
    cleanup_user_data(context)
    return await start(update, context)

# --- FEATURE 3: JSON DIVIDE ---
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
    progress_message = await update.message.reply_text("Zip file received. Dividing JSON and masking images with inpainting...")
    json_data = context.user_data.get('json_data')
    if not json_data:
        await progress_message.edit_text("Error: JSON data was lost. Please start over.")
        return await back_to_main_menu(update, context)

    with tempfile.TemporaryDirectory() as temp_dir:
        working_dir = Path(temp_dir) / "work"
        working_dir.mkdir()
        zip_tg_file = await update.message.document.get_file()
        zip_path = working_dir / "images.zip"
        await zip_tg_file.download_to_drive(zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(working_dir)
        
        blocks_by_folder = {}
        all_images_to_process = set()
        for entry in json_data:
            p = Path(entry['filename'])
            folder_name = str(p.parent)
            if folder_name not in blocks_by_folder: blocks_by_folder[folder_name] = []
            blocks_by_folder[folder_name].append(entry)
            # Add the full relative path to the set of images to process
            all_images_to_process.add(entry['filename'])

        total_images = len(all_images_to_process)
        processed_count = 0
        
        for folder_rel_path, blocks in blocks_by_folder.items():
            folder_abs_path = working_dir / Path(folder_rel_path)
            if not folder_abs_path.is_dir(): continue
            
            # Save the per-folder JSON file
            folder_json_path = folder_abs_path / "folder_text.json"
            with open(folder_json_path, 'w', encoding='utf-8') as f:
                json.dump(blocks, f, ensure_ascii=False, indent=4)
            
            # Identify unique image names within this folder's blocks
            images_in_folder = {Path(b['filename']).name for b in blocks}
            
            for img_name in images_in_folder:
                img_path_str = str(folder_abs_path / img_name)
                
                # --- NEW: INPAINTING LOGIC ---
                img_cv = cv2.imread(img_path_str)
                if img_cv is None:
                    logger.warning(f"Could not read image for masking: {img_path_str}")
                    continue

                # Create a mask for all text blocks on this specific image
                mask = np.zeros(img_cv.shape[:2], dtype=np.uint8)
                boxes_to_mask = [b['bbox'] for b in blocks if Path(b['filename']).name == img_name]
                for bbox_points in boxes_to_mask:
                    points = np.array(bbox_points, dtype=np.int32)
                    cv2.fillPoly(mask, [points], 255)

                # Inpaint the original image to remove text
                inpainted_img_cv = cv2.inpaint(img_cv, mask, 3, cv2.INPAINT_TELEA)
                
                # Save the inpainted image, overwriting the original in the temp folder
                cv2.imwrite(img_path_str, inpainted_img_cv)
                # --- END NEW ---

                processed_count +=1
                if processed_count % 5 == 0: # Update progress every 5 images
                    await send_progress_update(progress_message, processed_count, total_images, "Masking")

        zip_path_str = os.path.join(temp_dir, "final_divided_comics")
        shutil.make_archive(zip_path_str, 'zip', working_dir)
        await progress_message.delete()
        await update.message.reply_document(document=open(f"{zip_path_str}.zip", 'rb'), caption="Dividing and masking complete!")

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
                CallbackQueryHandler(json_maker_prompt_files, pattern="^lang_"),
                CallbackQueryHandler(back_to_main_menu, pattern="^main_menu_start$"),
            ],
            WAITING_FILES_OCR: [MessageHandler(filters.Document.ZIP, extract_text_from_files)],
            
            WAITING_JSON_TRANSLATE_ZIP: [MessageHandler(filters.Document.FileExtension("json"), json_translate_get_json)],
            WAITING_ZIP_TRANSLATE: [MessageHandler(filters.Document.ZIP, json_translate_process_zip)],

            WAITING_JSON_DIVIDE: [MessageHandler(filters.Document.FileExtension("json"), json_divide_get_json)],
            WAITING_ZIP_DIVIDE: [MessageHandler(filters.Document.ZIP, json_divide_process_zip)],
        },
        fallbacks=[CommandHandler("start", start)],
    )
    application.add_handler(conv_handler)
    application.run_polling()

if __name__ == "__main__":
    # Add filetype import at the top
    import filetype
    main()
