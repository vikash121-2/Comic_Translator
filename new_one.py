import logging
import os
import io
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
    JSON_TRANSLATE_CHOICE, 
    WAITING_JSON_TRANSLATE_IMG, WAITING_IMAGES_TRANSLATE,
    WAITING_JSON_TRANSLATE_ZIP, WAITING_ZIP_TRANSLATE,
    JSON_DIVIDE_CHOICE, 
    WAITING_JSON_DIVIDE, WAITING_ZIP_DIVIDE,
) = range(12)


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

def cleanup_user_data(context: ContextTypes.DEFAULT_TYPE):
    if 'temp_dir_obj' in context.user_data:
        context.user_data['temp_dir_obj'].cleanup()
        del context.user_data['temp_dir_obj']
    context.user_data.pop('image_paths', None)
    context.user_data.pop('received_images', None)
    context.user_data.pop('json_data', None)

def get_ocr_results(image_paths: List[str]) -> Dict:
    results = {}
    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        try:
            # Run all readers and combine results
            output_ja = reader_ja.readtext(image_path)
            output_ko = reader_ko.readtext(image_path)
            output_sim = reader_sim.readtext(image_path)
            output_tra = reader_tra.readtext(image_path)
            combined_output = output_ja + output_ko + output_sim + output_tra
            
            unique_results = {}
            for (bbox, text, prob) in combined_output:
                box_key = (text, int(bbox[0][1] / 10), int(bbox[0][0] / 10))
                if box_key not in unique_results:
                    unique_results[box_key] = (bbox, text, prob)

            text_blocks = []
            for (bbox, text, prob) in unique_results.values():
                x_coords = [int(p[0]) for p in bbox]
                y_coords = [int(p[1]) for p in bbox]
                simple_bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                text_blocks.append({"text": text, "location": simple_bbox})
            results[image_name] = text_blocks
        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {e}")
            results[image_name] = []
    return results

def sort_text_blocks(ocr_data: Dict) -> Dict:
    sorted_data = {"images": []}
    for image_info in ocr_data["images"]:
        sorted_blocks = sorted(image_info["text_blocks"], key=lambda b: (b["location"][1], b["location"][0]))
        sorted_data["images"].append({"image_name": image_info["image_name"], "text_blocks": sorted_blocks})
    return sorted_data

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

async def json_maker_prompt_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    context.user_data['temp_dir_obj'] = tempfile.TemporaryDirectory()
    context.user_data['image_paths'] = []
    await query.answer()
    await query.edit_message_text("Please send your images. Press 'Done Uploading' when finished.")
    return WAITING_IMAGES_OCR

async def collect_images(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        temp_dir_path = context.user_data['temp_dir_obj'].name
        image_paths = context.user_data['image_paths']
    except KeyError:
        await update.message.reply_text("Something went wrong. Please start over with /start.")
        cleanup_user_data(context)
        return ConversationHandler.END

    file_to_download, file_name = (None, None)
    if update.message.photo:
        photo = update.message.photo[-1]
        file_to_download = await photo.get_file()
        file_name = f"{photo.file_id}.jpg"
    elif update.message.document and update.message.document.mime_type.startswith('image/'):
        doc = update.message.document
        file_to_download = await doc.get_file()
        file_name = doc.file_name
    else:
        await update.message.reply_text("That doesn't seem to be an image. Please send a photo or image file.")
        return WAITING_IMAGES_OCR

    file_path = os.path.join(temp_dir_path, file_name)
    await file_to_download.download_to_drive(file_path)
    image_paths.append(file_path)
    keyboard = [[InlineKeyboardButton("✅ Done Uploading", callback_data="jm_process_images")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(f"Image {len(image_paths)} received. Send another, or press Done.", reply_markup=reply_markup)
    return WAITING_IMAGES_OCR

async def process_collected_images(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("Processing images...")
    
    image_paths = context.user_data.get('image_paths', [])
    if not image_paths:
        await query.edit_message_text("You didn't send any images! Please start over.")
        cleanup_user_data(context)
        return await start(update, context)

    ocr_data = get_ocr_results(image_paths)
    raw_json = {"images": [{"image_name": name, "text_blocks": blocks} for name, blocks in ocr_data.items()]}
    final_json = sort_text_blocks(raw_json)
    
    if sum(len(img["text_blocks"]) for img in final_json["images"]) == 0:
        await query.edit_message_text("I couldn't extract any text. Returning to the main menu.")
        cleanup_user_data(context)
        return await start(update, context)

    temp_dir_path = context.user_data['temp_dir_obj'].name
    json_path = os.path.join(temp_dir_path, "extracted_text.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(final_json, f, ensure_ascii=False, indent=4)
        
    await context.bot.send_document(chat_id=query.message.chat.id, document=open(json_path, 'rb'))
    cleanup_user_data(context)
    return await start(update, context)

async def json_maker_prompt_zip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("Please send the zip file.")
    return WAITING_ZIP_OCR

async def json_maker_process_zip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Processing zip file...")
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_file = await update.message.document.get_file()
        zip_path = os.path.join(temp_dir, "input.zip")
        await zip_file.download_to_drive(zip_path)
        extract_path = os.path.join(temp_dir, "extracted")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_path)
        
        final_json = {"folders": []}
        for folder_name in sorted(os.listdir(extract_path)):
            folder_path = os.path.join(extract_path, folder_name)
            if os.path.isdir(folder_path):
                image_paths = [os.path.join(folder_path, f) for f in sorted(os.listdir(folder_path)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if image_paths:
                    ocr_data = get_ocr_results(image_paths)
                    raw_json = {"images": [{"image_name": name, "text_blocks": blocks} for name, blocks in ocr_data.items()]}
                    sorted_json = sort_text_blocks(raw_json)
                    final_json["folders"].append({"folder_name": folder_name, "images": sorted_json["images"]})
        
        json_path = os.path.join(temp_dir, "extracted_text_from_zip.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(final_json, f, ensure_ascii=False, indent=4)
        await update.message.reply_document(document=open(json_path, 'rb'))
    return await start(update, context)

# --- 2. Json To Comic Translate Feature ---

async def json_translate_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    keyboard = [
        [InlineKeyboardButton("🖼️ Image Upload", callback_data="jt_image")],
        [InlineKeyboardButton("🗂️ Zip Upload", callback_data="jt_zip")],
        [InlineKeyboardButton("« Back", callback_data="main_menu_start")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.answer()
    await query.edit_message_text("Choose an option for translation:", reply_markup=reply_markup)
    return JSON_TRANSLATE_CHOICE

async def json_translate_prompt_json_for_img(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("First, please upload the JSON file with the translated text.")
    return WAITING_JSON_TRANSLATE_IMG

async def json_translate_get_json_for_img(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    json_file = await update.message.document.get_file()
    json_bytes = await json_file.download_as_bytearray()
    context.user_data['json_data'] = json.loads(json_bytes)
    
    context.user_data['temp_dir_obj'] = tempfile.TemporaryDirectory()
    context.user_data['received_images'] = {}
    
    await update.message.reply_text("JSON received. Now, please send the corresponding original images. Press 'Done' when finished.")
    return WAITING_IMAGES_TRANSLATE

async def json_translate_collect_images(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        temp_dir_path = context.user_data['temp_dir_obj'].name
        received_images = context.user_data['received_images']
    except KeyError:
        await update.message.reply_text("Something went wrong. Please start over with /start.")
        cleanup_user_data(context)
        return ConversationHandler.END

    file_to_download, original_filename = (None, None)
    if update.message.photo:
        photo = update.message.photo[-1]
        file_to_download = await photo.get_file()
        original_filename = f"{photo.file_id}.jpg"
    elif update.message.document and update.message.document.mime_type.startswith('image/'):
        doc = update.message.document
        file_to_download = await doc.get_file()
        original_filename = doc.file_name
    else:
        return WAITING_IMAGES_TRANSLATE

    file_path = os.path.join(temp_dir_path, original_filename)
    await file_to_download.download_to_drive(file_path)
    received_images[original_filename] = file_path

    keyboard = [[InlineKeyboardButton("✅ Done Uploading", callback_data="jt_process_images")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(f"Image '{original_filename}' received. Send another, or press Done.", reply_markup=reply_markup)
    return WAITING_IMAGES_TRANSLATE

async def json_translate_process_images(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Draws text on all collected images and sends them back."""
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("Processing translation for all images...")

    json_data = context.user_data.get('json_data', {})
    received_images = context.user_data.get('received_images', {})
    
    if not received_images:
        await query.edit_message_text("You didn't send any images! Please start over.")
        cleanup_user_data(context)
        return await start(update, context)

    try:
        # Using a default font size, can be adjusted
        font = ImageFont.truetype(FONT_PATH, 20)
    except IOError:
        await context.bot.send_message(chat_id=query.message.chat.id, text=f"⚠️ Error: Font file '{FONT_PATH}' not found.")
        cleanup_user_data(context)
        return await start(update, context)

    # Smartly find the list of images, whether the JSON is from a zip or single images
    images_json_data = []
    if "images" in json_data:
        images_json_data = json_data["images"]
    elif "folders" in json_data and len(json_data["folders"]) > 0:
        # If from a zip, just use the images from the first folder
        images_json_data = json_data["folders"][0].get("images", [])

    if not images_json_data:
        await query.edit_message_text("Could not find an 'images' list in your JSON file. Please check the file format.")
        cleanup_user_data(context)
        return await start(update, context)
    
    images_processed = 0
    for image_info in images_json_data:
        image_name = image_info.get("image_name")
        # Check if this image was one of the ones the user uploaded
        if image_name in received_images:
            image_path = received_images[image_name]
            
            with Image.open(image_path).convert("RGBA") as img:
                # Create a transparent layer to draw on
                txt_layer = Image.new("RGBA", img.size, (255, 255, 255, 0))
                draw = ImageDraw.Draw(txt_layer)
                
                for block in image_info.get("text_blocks", []):
                    text = block["text"]
                    loc = block["location"]
                    
                    # Draw a white rectangle to cover the old text, then the new text
                    draw.rectangle(loc, fill="white")
                    draw.text((loc[0], loc[1]), text, font=font, fill="black")
                
                # Composite the text layer onto the original image
                combined_img = Image.alpha_composite(img, txt_layer).convert("RGB")
                
                # Send the processed image back to the user
                bio = io.BytesIO()
                bio.name = f"translated_{image_name}"
                combined_img.save(bio, 'JPEG')
                bio.seek(0)
                await context.bot.send_document(chat_id=query.message.chat.id, document=bio)
                images_processed += 1

    if images_processed == 0:
        await context.bot.send_message(chat_id=query.message.chat.id, text="Warning: No matching images found between your JSON file and the images you uploaded. Please check the 'image_name' fields.")
    else:
        await context.bot.send_message(chat_id=query.message.chat.id, text="Translation complete!")
        
    cleanup_user_data(context)
    return await start(update, context)

async def json_translate_prompt_json_for_zip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("Please upload the JSON file that contains the translated text.")
    return WAITING_JSON_TRANSLATE_ZIP

async def json_translate_get_json_for_zip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    json_file = await update.message.document.get_file()
    json_bytes = await json_file.download_as_bytearray()
    context.user_data['json_data'] = json.loads(json_bytes)
    await update.message.reply_text("JSON file received. Now, please upload the original zip file with the images.")
    return WAITING_ZIP_TRANSLATE

async def json_translate_process_zip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # This is the full function from earlier, now integrated.
    await update.message.reply_text("Processing translation... This may take a while.")
    json_data = context.user_data['json_data']
    
    try: font = ImageFont.truetype(FONT_PATH, 20)
    except IOError:
        await update.message.reply_text(f"⚠️ Error: Font file '{FONT_PATH}' not found.")
        cleanup_user_data(context)
        return await start(update, context)

    with tempfile.TemporaryDirectory() as temp_dir:
        zip_file = await update.message.document.get_file()
        input_zip_path = os.path.join(temp_dir, "input.zip")
        await zip_file.download_to_drive(input_zip_path)
        extract_path = os.path.join(temp_dir, "extracted")
        output_path = os.path.join(temp_dir, "output")
        with zipfile.ZipFile(input_zip_path, 'r') as z: z.extractall(extract_path)

        for folder_data in json_data.get("folders", []):
            folder_name = folder_data["folder_name"]
            output_folder_path = os.path.join(output_path, folder_name)
            os.makedirs(output_folder_path, exist_ok=True)
            for image_data in folder_data.get("images", []):
                image_name = image_data["image_name"]
                original_image_path = os.path.join(extract_path, folder_name, image_name)
                if os.path.exists(original_image_path):
                    with Image.open(original_image_path).convert("RGB") as img:
                        draw = ImageDraw.Draw(img)
                        for block in image_data.get("text_blocks", []):
                            text, loc = block["text"], block["location"]
                            draw.rectangle(loc, fill="white")
                            draw.text((loc[0], loc[1]), text, font=font, fill="black")
                        img.save(os.path.join(output_folder_path, image_name))
        
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
        zip_file = await update.message.document.get_file()
        input_zip_path = os.path.join(temp_dir, "input.zip")
        await zip_file.download_to_drive(input_zip_path)
        extract_path = os.path.join(temp_dir, "extracted")
        with zipfile.ZipFile(input_zip_path, 'r') as z: z.extractall(extract_path)

        for folder_data in json_data.get("folders", []):
            folder_name = folder_data["folder_name"]
            current_folder_path = os.path.join(extract_path, folder_name)
            if os.path.isdir(current_folder_path):
                folder_specific_json = {"images": folder_data.get("images", [])}
                folder_json_path = os.path.join(current_folder_path, f"{folder_name}.json")
                with open(folder_json_path, 'w', encoding='utf-8') as f:
                    json.dump(folder_specific_json, f, ensure_ascii=False, indent=4)
                
                for image_data in folder_data.get("images", []):
                    image_name = image_data["image_name"]
                    image_path = os.path.join(current_folder_path, image_name)
                    if os.path.exists(image_path):
                        with Image.open(image_path).convert("RGB") as img:
                            draw = ImageDraw.Draw(img)
                            for block in image_data.get("text_blocks", []):
                                draw.rectangle(block["location"], fill="black")
                            img.save(image_path)
        
        output_zip_name = os.path.join(temp_dir, "divided_masked_comic")
        shutil.make_archive(output_zip_name, 'zip', extract_path)
        await update.message.reply_document(document=open(f"{output_zip_name}.zip", 'rb'))
        
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
                CallbackQueryHandler(json_maker_prompt_image, pattern="^jm_image$"),
                CallbackQueryHandler(json_maker_prompt_zip, pattern="^jm_zip$"),
                CallbackQueryHandler(back_to_main_menu, pattern="^main_menu_start$"),
            ],
            WAITING_IMAGES_OCR: [
                MessageHandler(filters.PHOTO | filters.Document.IMAGE, collect_images),
                CallbackQueryHandler(process_collected_images, pattern="^jm_process_images$"),
            ],
            WAITING_ZIP_OCR: [MessageHandler(filters.Document.ZIP, json_maker_process_zip)],
            # States for JSON Translate
            JSON_TRANSLATE_CHOICE: [
                CallbackQueryHandler(json_translate_prompt_json_for_img, pattern="^jt_image$"),
                CallbackQueryHandler(json_translate_prompt_json_for_zip, pattern="^jt_zip$"),
                CallbackQueryHandler(back_to_main_menu, pattern="^main_menu_start$")
            ],
            WAITING_JSON_TRANSLATE_IMG: [MessageHandler(filters.Document.FileExtension("json"), json_translate_get_json_for_img)],
            WAITING_IMAGES_TRANSLATE: [
                MessageHandler(filters.PHOTO | filters.Document.IMAGE, json_translate_collect_images),
                CallbackQueryHandler(json_translate_process_images, pattern="^jt_process_images$"),
            ],
            WAITING_JSON_TRANSLATE_ZIP: [MessageHandler(filters.Document.FileExtension("json"), json_translate_get_json_for_zip)],
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
