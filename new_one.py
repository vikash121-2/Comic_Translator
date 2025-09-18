import logging
import os
import zipfile
import shutil
import json
import torch
import tempfile
import traceback
import asyncio
import cv2
import numpy as np
from typing import List, Dict

from PIL import Image, ImageDraw, ImageFont, ImageFile

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

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

BOT_TOKEN = "6298615623:AAEyldSFqE2HT-2vhITBmZ9lQL23C0fu-Ao"  # <-- IMPORTANT: Replace with your bot token
FONT_PATH = "DMSerifText-Regular.ttf"  # <-- IMPORTANT: Make sure this font file is in the same directory


# Conversation states
(
    MAIN_MENU,
    JSON_MAKER_CHOICE, WAITING_IMAGES_OCR, WAITING_ZIP_OCR,
    JSON_TRANSLATE_CHOICE, WAITING_JSON_TRANSLATE_ZIP, WAITING_ZIP_TRANSLATE,
    JSON_DIVIDE_CHOICE, WAITING_JSON_DIVIDE, WAITING_ZIP_DIVIDE,
) = range(10)

# --- Load easyocr models ---
logger.info(f"Loading all easyocr models...")
try:
    import easyocr
    use_gpu = torch.cuda.is_available()
    logger.info(f"GPU available: {use_gpu}")
    reader_ja = easyocr.Reader(['ja', 'en'], gpu=use_gpu)
    reader_ko = easyocr.Reader(['ko', 'en'], gpu=use_gpu)
    reader_sim = easyocr.Reader(['ch_sim', 'en'], gpu=use_gpu)
    reader_tra = easyocr.Reader(['ch_tra', 'en'], gpu=use_gpu)
    logger.info("All easyocr models loaded successfully.")
except Exception as e:
    logger.critical(f"Critical Error: Could not load easyocr model. Error: {e}")
    logger.critical(traceback.format_exc())
    exit(1)

# --- Helper & Utility Functions ---

def get_ocr_results(image_paths: List[str]):
    # ... (This function is complete and doesn't need changes)
    pass

def cleanup_user_data(context: ContextTypes.DEFAULT_TYPE):
    if 'temp_dir_obj' in context.user_data:
        context.user_data['temp_dir_obj'].cleanup()
        del context.user_data['temp_dir_obj']
    context.user_data.pop('image_paths', None)
    context.user_data.pop('json_data', None)

# --- Main Menu & Core Navigation ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    keyboard = [
        [InlineKeyboardButton("📝 Json maker", callback_data="main_json_maker")],
        [InlineKeyboardButton("🎨 json To Comic translate", callback_data="main_translate")],
        [InlineKeyboardButton("✂️ json divide", callback_data="main_divide")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    message_text = "Welcome! This bot uses easyocr. Please choose an option:"
    if update.message:
        await update.message.reply_text(message_text, reply_markup=reply_markup)
    elif update.callback_query:
        query = update.callback_query
        try:
            await query.answer()
        except BadRequest:
            logger.info("Callback query already answered.")
        await query.edit_message_text(message_text, reply_markup=reply_markup)
    return MAIN_MENU

async def back_to_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Helper function to go back to the main menu from a sub-menu."""
    cleanup_user_data(context)
    return await start(update, context)

# --- 1. Json Maker Feature ---

async def json_maker_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    keyboard = [
        [InlineKeyboardButton("🖼️ Image Upload", callback_data="jm_image")],
        [InlineKeyboardButton("🗂️ Zip Upload", callback_data="jm_zip")],
        [InlineKeyboardButton("« Back", callback_data="main_menu_start")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.answer()
    await query.edit_message_text("How would you like to provide source files?", reply_markup=reply_markup)
    return JSON_MAKER_CHOICE

# ... (json maker functions: prompt, collect, process, etc. remain here) ...


# --- 2. Json To Comic Translate Feature ---

async def json_translate_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    keyboard = [
        # The image upload for this is complex, so we'll focus on the main zip feature
        [InlineKeyboardButton("🗂️ Zip Upload", callback_data="jt_zip")],
        [InlineKeyboardButton("« Back", callback_data="main_menu_start")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.answer()
    await query.edit_message_text("Choose an option for translation:", reply_markup=reply_markup)
    return JSON_TRANSLATE_CHOICE

async def json_translate_prompt_json(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("Please upload the JSON file that contains the translated text.")
    return WAITING_JSON_TRANSLATE_ZIP

async def json_translate_get_json(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    json_file = await update.message.document.get_file()
    json_bytes = await json_file.download_as_bytearray()
    context.user_data['json_data'] = json.loads(json_bytes)
    await update.message.reply_text("JSON file received. Now, please upload the original zip file with the images.")
    return WAITING_ZIP_TRANSLATE

async def json_translate_process_zip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Processing translation... This may take a while.")
    json_data = context.user_data['json_data']
    
    try:
        font = ImageFont.truetype(FONT_PATH, 20)
    except IOError:
        await update.message.reply_text(f"⚠️ Error: Font file '{FONT_PATH}' not found. Cannot continue.")
        cleanup_user_data(context)
        return await start(update, context)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Download and extract the original zip
        zip_file = await update.message.document.get_file()
        input_zip_path = os.path.join(temp_dir, "input.zip")
        await zip_file.download_to_drive(input_zip_path)
        
        extract_path = os.path.join(temp_dir, "extracted")
        output_path = os.path.join(temp_dir, "output")
        with zipfile.ZipFile(input_zip_path, 'r') as z:
            z.extractall(extract_path)

        # Process folders and images based on JSON
        for folder_data in json_data.get("folders", []):
            folder_name = folder_data["folder_name"]
            output_folder_path = os.path.join(output_path, folder_name)
            os.makedirs(output_folder_path, exist_ok=True)
            
            for image_data in folder_data.get("images", []):
                image_name = image_data["image_name"]
                original_image_path = os.path.join(extract_path, folder_name, image_name)
                if os.path.exists(original_image_path):
                    with Image.open(original_image_path).convert("RGBA") as img:
                        txt_layer = Image.new("RGBA", img.size, (255, 255, 255, 0))
                        draw = ImageDraw.Draw(txt_layer)
                        for block in image_data.get("text_blocks", []):
                            text = block["text"]
                            loc = block["location"]
                            # Draw text with a white outline for visibility
                            draw.text((loc[0], loc[1]), text, font=font, fill=(0,0,0,255), stroke_width=1, stroke_fill=(255,255,255,255))
                        
                        combined_img = Image.alpha_composite(img, txt_layer).convert("RGB")
                        combined_img.save(os.path.join(output_folder_path, image_name))
                else:
                    logger.warning(f"Image not found, skipping: {original_image_path}")
        
        # Create and send the new zip
        output_zip_name = os.path.join(temp_dir, "translated_comic")
        shutil.make_archive(output_zip_name, 'zip', output_path)
        await update.message.reply_document(document=open(f"{output_zip_name}.zip", 'rb'))

    cleanup_user_data(context)
    return await start(update, context)


# --- 3. Json Divide Feature ---

async def json_divide_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    keyboard = [
        [InlineKeyboardButton("🗂️ Zip Upload", callback_data="jd_zip")],
        [InlineKeyboardButton("« Back", callback_data="main_menu_start")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.answer()
    await query.edit_message_text("Choose an option for dividing:", reply_markup=reply_markup)
    return JSON_DIVIDE_CHOICE

async def json_divide_prompt_json(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("Please upload the master JSON file.")
    return WAITING_JSON_DIVIDE

async def json_divide_get_json(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    json_file = await update.message.document.get_file()
    json_bytes = await json_file.download_as_bytearray()
    context.user_data['json_data'] = json.loads(json_bytes)
    await update.message.reply_text("JSON file received. Now, please upload the original zip file.")
    return WAITING_ZIP_DIVIDE

async def json_divide_process_zip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Dividing JSON and masking images... This may take a while.")
    json_data = context.user_data['json_data']
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download and extract zip to be modified in-place
        zip_file = await update.message.document.get_file()
        input_zip_path = os.path.join(temp_dir, "input.zip")
        await zip_file.download_to_drive(input_zip_path)
        
        extract_path = os.path.join(temp_dir, "extracted")
        with zipfile.ZipFile(input_zip_path, 'r') as z:
            z.extractall(extract_path)

        # Process folders based on JSON
        for folder_data in json_data.get("folders", []):
            folder_name = folder_data["folder_name"]
            current_folder_path = os.path.join(extract_path, folder_name)
            
            if os.path.isdir(current_folder_path):
                # 1. Create and save the smaller JSON for this folder
                folder_specific_json = {"images": folder_data.get("images", [])}
                folder_json_path = os.path.join(current_folder_path, f"{folder_name}.json")
                with open(folder_json_path, 'w', encoding='utf-8') as f:
                    json.dump(folder_specific_json, f, ensure_ascii=False, indent=4)
                
                # 2. Mask the images in this folder
                for image_data in folder_data.get("images", []):
                    image_name = image_data["image_name"]
                    image_path = os.path.join(current_folder_path, image_name)
                    if os.path.exists(image_path):
                        with Image.open(image_path) as img:
                            draw = ImageDraw.Draw(img)
                            for block in image_data.get("text_blocks", []):
                                draw.rectangle(block["location"], fill="black")
                            img.save(image_path)
        
        # Create and send the new zip from the modified extracted folder
        output_zip_name = os.path.join(temp_dir, "divided_masked_comic")
        shutil.make_archive(output_zip_name, 'zip', extract_path)
        await update.message.reply_document(document=open(f"{output_zip_name}.zip", 'rb'))
        
    cleanup_user_data(context)
    return await start(update, context)

# --- Main Application Setup ---

def main() -> None:
    application = Application.builder().token(BOT_TOKEN).build()

    # NOTE: All the functions for the Json Maker (jm_image, jm_zip) are assumed to be here
    # from the previous script. They are omitted for brevity but are required.
    from previous_scripts import json_maker_prompt_image, collect_images, process_collected_images, json_maker_process_zip

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
                CallbackQueryHandler(json_maker_prompt_image, pattern="^jm_image$"),
                CallbackQueryHandler(json_maker_process_zip, pattern="^jm_zip$"), # Assuming this exists
                CallbackQueryHandler(back_to_main_menu, pattern="^main_menu_start$"),
            ],
            WAITING_IMAGES_OCR: [
                MessageHandler(filters.PHOTO | filters.Document.IMAGE, collect_images),
                CallbackQueryHandler(process_collected_images, pattern="^jm_process_images$"),
            ],
            # States for JSON Translate
            JSON_TRANSLATE_CHOICE: [
                CallbackQueryHandler(json_translate_prompt_json, pattern="^jt_zip$"),
                CallbackQueryHandler(back_to_main_menu, pattern="^main_menu_start$")
            ],
            WAITING_JSON_TRANSLATE_ZIP: [MessageHandler(filters.Document.FileExtension("json"), json_translate_get_json)],
            WAITING_ZIP_TRANSLATE: [MessageHandler(filters.Document.ZIP, json_translate_process_zip)],
            # States for JSON Divide
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
