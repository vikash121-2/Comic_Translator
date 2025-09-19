import logging
import os
import io
import zipfile
import shutil
import json
import torch
import tempfile
import traceback
import textwrap
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

BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
FONT_PATH = "font.ttf"

# Conversation states
(
    MAIN_MENU,
    JSON_MAKER_CHOICE, WAITING_IMAGES_OCR, SELECT_LANGUAGE, WAITING_ZIP_OCR,
    JSON_TRANSLATE_CHOICE,
    WAITING_JSON_TRANSLATE_IMG, WAITING_IMAGES_TRANSLATE,
    WAITING_JSON_TRANSLATE_ZIP, WAITING_ZIP_TRANSLATE,
    JSON_DIVIDE_CHOICE,
    WAITING_JSON_DIVIDE, WAITING_ZIP_DIVIDE,
) = range(13)

# --- Load Surya-OCR model ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Loading Surya OCR model onto {DEVICE}...")
try:
    from surya.ocr import run_ocr
    from surya.model.detection import segformer
    from surya.model.recognition import vit
    det_processor, det_model = segformer.load_processor_and_model()
    rec_processor, rec_model = vit.load_processor_and_model()
    logger.info("Surya OCR model loaded successfully.")
except Exception as e:
    logger.critical(f"Critical Error: Could not load Surya OCR model. Error: {e}")
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
    context.user_data.pop('zip_file', None)

def get_ocr_results(image_paths: List[str], lang_code: str) -> Dict:
    results = {}
    try:
        images = [Image.open(p).convert("RGB") for p in image_paths]
        langs = [[lang_code]] * len(images)
        predictions = run_ocr(images, langs, det_model, det_processor, rec_model, rec_processor)

        for i, (pred, image_path) in enumerate(zip(predictions, image_paths)):
            image_name = os.path.basename(image_path)
            text_blocks = [{"text": line.text, "location": line.bbox} for line in pred.text_lines]
            results[image_name] = text_blocks
    except Exception as e:
        logger.error(f"Failed during Surya OCR processing: {e}")
        for image_path in image_paths:
            results[os.path.basename(image_path)] = []
    return results

def draw_text_in_box(draw: ImageDraw, box: List[int], text: str, font_path: str, max_font_size: int = 60):
    """
    Draws wrapped, centered, and auto-sized text inside a given bounding box.
    """
    box_width = box[2] - box[0]
    box_height = box[3] - box[1]

    if box_width <= 0 or box_height <= 0:
        return

    # Find the best font size by starting large and decreasing
    font_size = max_font_size
    font = ImageFont.truetype(font_path, font_size)
    
    while font_size > 5:
        # Estimate how many characters can fit on a line
        avg_char_width = font.getlength("a") 
        wrap_width = max(1, int(box_width / avg_char_width))
        
        wrapped_text = textwrap.wrap(text, width=wrap_width, break_long_words=True)
        
        # Calculate the total height of the wrapped text block
        total_text_height = 0
        line_heights = []
        for line in wrapped_text:
            try:
                line_bbox = draw.textbbox((0, 0), line, font=font)
                line_heights.append(line_bbox[3] - line_bbox[1])
            except (ValueError, TypeError): # Handle potential empty lines
                line_heights.append(font_size)

        total_text_height = sum(line_heights)

        # Check if the text block fits within the box
        if total_text_height <= box_height:
            break
        font_size -= 2 # Decrease font size and try again
        font = ImageFont.truetype(font_path, font_size)
    
    # Draw the final, sized, and wrapped text
    y_start = box[1] + (box_height - total_text_height) / 2  # Center vertically
    for i, line in enumerate(wrapped_text):
        line_width = font.getlength(line)
        x_start = box[0] + (box_width - line_width) / 2 # Center horizontally
        draw.text((x_start, y_start), line, font=font, fill="black")
        y_start += line_heights[i]


# --- Main Menu & Core Navigation ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # ... (code is the same as before)
    pass

async def back_to_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # ... (code is the same as before)
    pass


# --- 1. Json Maker Feature ---

async def json_maker_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # ... (code is the same as before)
    pass

async def json_maker_prompt_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # ... (code is the same as before)
    pass

async def collect_images(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # ... (code is the same as before)
    pass

async def prompt_language(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    keyboard = [
        [InlineKeyboardButton("Japanese", callback_data="lang_ja"), InlineKeyboardButton("Korean", callback_data="lang_ko")],
        [InlineKeyboardButton("Chinese", callback_data="lang_ch"), InlineKeyboardButton("English", callback_data="lang_en")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.answer()
    await query.edit_message_text("Please select the primary language of the image(s):", reply_markup=reply_markup)
    return SELECT_LANGUAGE

async def process_images_with_selected_language(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    lang_code = query.data.split('_', 1)[1]
    await query.answer()
    await query.edit_message_text(f"Processing images with Surya OCR ({lang_code.upper()})...")
    
    # ... (rest of the function is the same, just ensure it doesn't have old `sort_text_blocks` calls if not needed)
    pass

async def json_maker_prompt_zip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # ... (code is the same as before)
    pass

async def process_zip_with_selected_language(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # ... (code is the same as before)
    pass


# --- 2. Json To Comic Translate Feature ---

async def json_translate_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # ... (code is the same as before)
    pass

async def json_translate_process_zip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Applying translation...")
    json_data = context.user_data['json_data']
    with tempfile.TemporaryDirectory() as temp_dir:
        # ... (zip extraction logic is the same) ...
        for folder_data in json_data.get("folders", []):
            # ...
            for image_data in folder_data.get("images", []):
                # ...
                if os.path.exists(original_image_path):
                    with Image.open(original_image_path).convert("RGB") as img:
                        draw = ImageDraw.Draw(img)
                        for block in image_data.get("text_blocks", []):
                            text, loc = block["text"], block["location"]
                            draw.rectangle(loc, fill="white")
                            draw_text_in_box(draw, loc, text, FONT_PATH) # Use new function
                        img.save(os.path.join(output_folder_path, image_name))
        # ... (re-zipping and sending is the same) ...
    cleanup_user_data(context)
    return await start(update, context)

async def json_translate_process_images(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # ... (setup is the same) ...
    for image_info in images_json_data:
        image_name = image_info.get("image_name")
        if image_name in received_images:
            image_path = received_images[image_name]
            with Image.open(image_path).convert("RGB") as img:
                draw = ImageDraw.Draw(img)
                for block in image_info.get("text_blocks", []):
                    text, loc = block["text"], block["location"]
                    draw.rectangle(loc, fill="white")
                    draw_text_in_box(draw, loc, text, FONT_PATH) # Use new function
                
                # ... (sending logic is the same) ...
    # ... (cleanup is the same) ...
    return await start(update, context)


# --- 3. Json Divide Feature ---
# ... (all functions for this feature remain the same) ...


# --- Main Application Setup ---

def main() -> None:
    # The full, final ConversationHandler setup goes here, unchanged from the last complete version.
    pass

if __name__ == "__main__":
    # To avoid making this response excessively long, I'm providing the key new and changed functions.
    # You must integrate these into the last complete script I provided.
    # Key changes are:
    # 1. New installation command.
    # 2. Switch OCR model loading to Surya-OCR.
    # 3. Replace the old `get_ocr_results` with the new Surya-based one.
    # 4. Add the new `draw_text_in_box` helper function.
    # 5. Replace the drawing logic in `json_translate_process_images` and `json_translate_process_zip`.
    # 6. Update the language selection menu for Surya's language codes ('ch' instead of 'ch_sim'/'ch_tra').
    pass
