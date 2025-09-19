import logging
import os
import io
import zipfile
import shutil
import json
import torch
import tempfile
import textwrap
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
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
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

# --- Load AI Models ---
logger.info(f"Loading all AI models...")
try:
    import easyocr
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    use_gpu = torch.cuda.is_available()
    device = "cuda:0" if use_gpu else "cpu"
    logger.info(f"Using device: {device}")

    # 1. Load easyocr for Text Detection
    detector = easyocr.Reader(['en'], gpu=use_gpu)
    logger.info("EasyOCR text detector loaded successfully.")

    # 2. Load TrOCR for high-accuracy Text Recognition
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed').to(device)
    logger.info("TrOCR text recognizer loaded successfully.")
except Exception as e:
    logger.critical(f"Critical Error: Could not load AI models. Error: {e}")
    exit(1)

# --- Helper & Utility Functions ---

def cleanup_user_data(context: ContextTypes.DEFAULT_TYPE):
    if 'temp_dir_obj' in context.user_data:
        context.user_data['temp_dir_obj'].cleanup()
    context.user_data.clear()

def get_ocr_results(image_paths: List[str]) -> Dict:
    results = {}
    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        try:
            image = Image.open(image_path).convert("RGB")
            
            # Use a more robust detection method from easyocr
            detected_results = detector.readtext(np.array(image), detail=1, paragraph=False)
            if not detected_results:
                results[image_name] = []
                continue

            text_blocks = []
            for (box_points, text, prob) in detected_results:
                x_coords = [p[0] for p in box_points]
                y_coords = [p[1] for p in box_points]
                simple_bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                
                # Crop the detected box and recognize text with TrOCR for higher accuracy
                cropped_image = image.crop(simple_bbox)
                pixel_values = processor(images=cropped_image, return_tensors="pt").pixel_values.to(device)
                generated_ids = model.generate(pixel_values)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                text_blocks.append({"text": generated_text, "location": simple_bbox})

            results[image_name] = text_blocks
            logger.info(f"Successfully processed {image_name} with Hybrid OCR.")
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
    
def draw_text_in_box(draw: ImageDraw, box: List[int], text: str, font_path: str, max_font_size: int = 60):
    box_width = box[2] - box[0]
    box_height = box[3] - box[1]
    if box_width <= 0 or box_height <= 0: return

    font_size = max_font_size
    font = ImageFont.truetype(font_path, font_size)
    
    while font_size > 5:
        avg_char_width = font.getlength("a") 
        wrap_width = max(1, int(box_width / avg_char_width))
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
        draw.text((x_start, y_start), line, font=font, fill="black")
        y_start += line_heights[i]

# --- Main Menu & Core Navigation ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    keyboard = [
        [InlineKeyboardButton("📝 Json maker", callback_data="main_json_maker")],
        [InlineKeyboardButton("🎨 json To Comic translate", callback_data="main_translate")],
        [InlineKeyboardButton("✂️ json divide", callback_data="main_divide")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    message_text = "Welcome! This bot uses a high-accuracy OCR model. Please choose an option:"
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

# --- All Feature Functions are included below ---

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
    await query.edit_message_text("Please send your images. Press 'Done' when finished.")
    return WAITING_IMAGES_OCR

async def collect_images(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        temp_dir_path = context.user_data['temp_dir_obj'].name
        image_paths = context.user_data['image_paths']
    except KeyError:
        await update.message.reply_text("Something went wrong. Please start over.")
        cleanup_user_data(context)
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
    keyboard = [[InlineKeyboardButton("✅ Done Uploading", callback_data="jm_process_images")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(f"Image {len(image_paths)} received. Send another, or press Done.", reply_markup=reply_markup)
    return WAITING_IMAGES_OCR

async def process_collected_images(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("Processing images with Hybrid OCR...")
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
    with open(json_path, 'w', encoding='utf-8') as f: json.dump(final_json, f, ensure_ascii=False, indent=4)
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
        with zipfile.ZipFile(zip_path, 'r') as z: z.extractall(extract_path)
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
        with open(json_path, 'w', encoding='utf-8') as f: json.dump(final_json, f, ensure_ascii=False, indent=4)
        await update.message.reply_document(document=open(json_path, 'rb'))
    return await start(update, context)

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
    context.user_data['json_data'] = json.loads(await json_file.download_as_bytearray())
    context.user_data['temp_dir_obj'] = tempfile.TemporaryDirectory()
    context.user_data['received_images'] = {}
    await update.message.reply_text("JSON received. Now, send the original images. Press 'Done' when finished.")
    return WAITING_IMAGES_TRANSLATE

async def json_translate_collect_images(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        temp_dir_path = context.user_data['temp_dir_obj'].name
        received_images = context.user_data['received_images']
    except KeyError:
        await update.message.reply_text("Something went wrong. Please start over.")
        cleanup_user_data(context)
        return ConversationHandler.END
    file_to_download, original_filename = (None, None)
    if update.message.photo:
        file_to_download = await update.message.photo[-1].get_file()
        original_filename = f"{file_to_download.file_id}.jpg"
    elif update.message.document and update.message.document.mime_type.startswith('image/'):
        file_to_download = await update.message.document.get_file()
        original_filename = update.message.document.file_name
    else: return WAITING_IMAGES_TRANSLATE
    file_path = os.path.join(temp_dir_path, original_filename)
    await file_to_download.download_to_drive(file_path)
    received_images[original_filename] = file_path
    keyboard = [[InlineKeyboardButton("✅ Done Uploading", callback_data="jt_process_images")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(f"Image '{original_filename}' received. Send another, or press Done.", reply_markup=reply_markup)
    return WAITING_IMAGES_TRANSLATE

async def json_translate_process_images(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("Applying translation to all images...")
    json_data, received_images = context.user_data.get('json_data', {}), context.user_data.get('received_images', {})
    if not received_images:
        await query.edit_message_text("You didn't send any images! Please start over.")
        cleanup_user_data(context)
        return await start(update, context)
    images_json_data = json_data.get("images", [])
    if "folders" in json_data and json_data["folders"]: images_json_data.extend(json_data["folders"][0].get("images", []))
    if not images_json_data:
        await query.edit_message_text("Could not find an 'images' list in your JSON file.")
        cleanup_user_data(context)
        return await start(update, context)
    images_processed = 0
    for image_info in images_json_data:
        image_name = image_info.get("image_name")
        if image_name in received_images:
            image_path = received_images[image_name]
            with Image.open(image_path).convert("RGB") as img:
                draw = ImageDraw.Draw(img)
                for block in image_info.get("text_blocks", []):
                    text, loc = block["text"], block["location"]
                    draw.rectangle(loc, fill="white")
                    draw_text_in_box(draw, loc, text, FONT_PATH)
                bio = io.BytesIO()
                bio.name = f"translated_{image_name}"
                img.save(bio, 'JPEG')
                bio.seek(0)
                await context.bot.send_document(chat_id=query.message.chat.id, document=bio)
                images_processed += 1
    if images_processed == 0:
        await context.bot.send_message(chat_id=query.message.chat.id, text="Warning: No matching image names found.")
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
    context.user_data['json_data'] = json.loads(await json_file.download_as_bytearray())
    await update.message.reply_text("JSON file received. Now, please upload the original zip file.")
    return WAITING_ZIP_TRANSLATE

async def json_translate_process_zip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Applying translation...")
    json_data = context.user_data['json_data']
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
                            draw_text_in_box(draw, loc, text, FONT_PATH)
                        img.save(os.path.join(output_folder_path, image_name))
        output_zip_name = os.path.join(temp_dir, "translated_comic")
        shutil.make_archive(output_zip_name, 'zip', output_path)
        await update.message.reply_document(document=open(f"{output_zip_name}.zip", 'rb'))
    cleanup_user_data(context)
    return await start(update, context)

async def json_divide_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    keyboard = [
        [InlineKeyboardButton("🗂️ Zip Upload", callback_data="jd_zip")],
        [InlineKeyboardButton("« Back", callback_data="main_menu_start")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.answer()
    await query.edit_message_text("This feature takes a master JSON and a zip file, then creates a divided zip with masked images.", reply_markup=reply_markup)
    return JSON_DIVIDE_CHOICE

async def json_divide_prompt_json(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("Please upload the master JSON file.")
    return WAITING_JSON_DIVIDE

async def json_divide_get_json(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    json_file = await update.message.document.get_file()
    context.user_data['json_data'] = json.loads(await json_file.download_as_bytearray())
    await update.message.reply_text("JSON file received. Now, please upload the original zip file.")
    return WAITING_ZIP_DIVIDE

async def json_divide_process_zip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Dividing JSON and masking images...")
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
                with open(folder_json_path, 'w', encoding='utf-8') as f: json.dump(folder_specific_json, f, ensure_ascii=False, indent=4)
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

# This is the crucial part that was missing before
if __name__ == "__main__":
    main()
