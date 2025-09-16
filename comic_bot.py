import logging
import os
import zipfile
import shutil
import json
import torch
import tempfile
from typing import List, Dict

from PIL import Image, ImageDraw, ImageFont

from telegram import (
    Update,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
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

# --- NEW: Transformers & Florence-2 OCR Initialization ---
try:
    from transformers import AutoProcessor, AutoModelForCausalLM
except ImportError:
    print("Error: Transformers library not found.")
    print("Please install it with: pip install transformers accelerate")
    exit(1)

# --- Basic Setup ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Global Variables & Configuration ---
BOT_TOKEN = "6298615623:AAEyldSFqE2HT-2vhITBmZ9lQL23C0fu-Ao"  # <-- IMPORTANT: Replace with your bot token
FONT_PATH = "DMSerifText-Regular.ttf"  # <-- IMPORTANT: Make sure this font file is in the same directory

# Conversation states
(
    MAIN_MENU,
    JSON_MAKER_CHOICE, WAITING_IMAGES_OCR, WAITING_ZIP_OCR,
    JSON_TRANSLATE_CHOICE, WAITING_JSON_TRANSLATE_IMG, WAITING_IMAGES_TRANSLATE,
    WAITING_JSON_TRANSLATE_ZIP, WAITING_ZIP_TRANSLATE,
    JSON_DIVIDE_CHOICE, WAITING_JSON_DIVIDE, WAITING_ZIP_DIVIDE,
    CHOOSE_LANG
) = range(13)

# --- NEW: Load Florence-2 model and processor once globally ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "microsoft/Florence-2-large"

logger.info(f"Loading Florence-2 model ({MODEL_ID}) onto {DEVICE}...")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, attn_implementation="eager").to(DEVICE)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
logger.info("Florence-2 model loaded successfully.")


# --- Helper Functions ---
def get_ocr_results(image_paths: List[str], lang: List[str] = None) -> Dict:
    """
    Runs OCR on a list of images using Florence-2 and returns a structured dictionary.
    The `lang` parameter is ignored as Florence-2 is multilingual.
    """
    results = {}
    task_prompt = "<OCR_WITH_REGION>"

    for image_path in image_paths:
        try:
            image = Image.open(image_path).convert("RGB")
            image_name = os.path.basename(image_path)

            # Prepare inputs for the model
            inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(DEVICE)

            # Generate text and bounding boxes
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False
            )
            
            # Decode the generated ids
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

            # Post-process to get structured output
            parsed_answer = processor.post_process_generation(
                generated_text, task=task_prompt, image_sizes=[image.size]
            )

            # Convert to the required JSON structure
            ocr_data = parsed_answer[task_prompt]
            text_blocks = []
            for bbox, label in zip(ocr_data['bboxes'], ocr_data['labels']):
                # The bbox is [xmin, ymin, xmax, ymax]
                text_blocks.append({
                    "text": label,
                    "location": bbox
                })
            
            results[image_name] = text_blocks
            logger.info(f"Successfully processed {image_name}")

        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {e}")
            results[os.path.basename(image_path)] = [] # Return empty list on failure

    return results

# --- Bot Command Handlers (No changes from here onwards, except for language handler) ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Starts the conversation and displays the main menu."""
    reply_keyboard = [
        ["Json maker"],
        ["json To Comic translate"],
        ["json divide"],
        ["Choose ocr language"],
    ]
    await update.message.reply_text(
        "Welcome! I use the Florence-2 model for OCR tasks.\n\n"
        "Please choose an option:",
        reply_markup=ReplyKeyboardMarkup(
            reply_keyboard, one_time_keyboard=True, resize_keyboard=True
        ),
    )
    return MAIN_MENU


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancels and ends the conversation."""
    await update.message.reply_text(
        "Operation cancelled.", reply_markup=ReplyKeyboardRemove()
    )
    # Clean up any stored data
    for key in ['media_group', 'json_data']:
        if key in context.user_data:
            del context.user_data[key]
    return ConversationHandler.END


# --- 1. JSON Maker ---
async def json_maker_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    reply_keyboard = [["image upload", "zip upload"]]
    await update.message.reply_text(
        "How would you like to provide the source files?",
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True, resize_keyboard=True),
    )
    return JSON_MAKER_CHOICE

async def json_maker_prompt_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Please send me one or more images.", reply_markup=ReplyKeyboardRemove())
    context.user_data['media_group'] = []
    return WAITING_IMAGES_OCR

async def json_maker_process_images(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Processes uploaded images, performs OCR, and sends a JSON file."""
    await update.message.reply_text("Processing images with Florence-2... This may take a moment.")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        image_paths = []
        # Use a simplified approach for single/multiple images for robustness
        messages = context.user_data.get('media_group', [update.message]) if 'media_group' in context.user_data and context.user_data['media_group'] else [update.message]

        for msg in messages:
            if msg.photo:
                photo_file = await msg.photo[-1].get_file()
                file_path = os.path.join(temp_dir, f"{photo_file.file_id}.jpg")
                await photo_file.download_to_drive(file_path)
                image_paths.append(file_path)

        ocr_data = get_ocr_results(image_paths)

        final_json = {
            "images": [
                {"image_name": name, "text_blocks": blocks}
                for name, blocks in ocr_data.items()
            ]
        }
        
        json_path = os.path.join(temp_dir, "extracted_text.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(final_json, f, ensure_ascii=False, indent=4)
            
        await update.message.reply_document(document=open(json_path, 'rb'))
    
    context.user_data.pop('media_group', None)
    await update.message.reply_text("Done! What would you like to do next? /start")
    return ConversationHandler.END

async def json_maker_prompt_zip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Please send me a zip file.", reply_markup=ReplyKeyboardRemove())
    return WAITING_ZIP_OCR

async def json_maker_process_zip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    zip_file = await update.message.document.get_file()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, "input.zip")
        await zip_file.download_to_drive(zip_path)
        
        extract_path = os.path.join(temp_dir, "extracted")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
            
        await update.message.reply_text("Zip file extracted. Starting OCR on all images...")
        
        final_json = {"folders": []}

        for folder_name in sorted(os.listdir(extract_path)):
            folder_path = os.path.join(extract_path, folder_name)
            if os.path.isdir(folder_path):
                folder_data = {"folder_name": folder_name, "images": []}
                image_paths = [os.path.join(folder_path, f) for f in sorted(os.listdir(folder_path)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                if image_paths:
                    ocr_data = get_ocr_results(image_paths)
                    for name, blocks in ocr_data.items():
                        folder_data["images"].append({"image_name": name, "text_blocks": blocks})
                
                final_json["folders"].append(folder_data)
        
        json_path = os.path.join(temp_dir, "extracted_text_from_zip.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(final_json, f, ensure_ascii=False, indent=4)
        
        await update.message.reply_document(document=open(json_path, 'rb'))

    await update.message.reply_text("Done! What would you like to do next? /start")
    return ConversationHandler.END


# --- 2. JSON to Comic Translate ---
async def json_translate_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    reply_keyboard = [["image upload", "zip upload"]]
    await update.message.reply_text(
        "How would you like to provide the files for translation?",
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True, resize_keyboard=True),
    )
    return JSON_TRANSLATE_CHOICE

async def json_translate_prompt_json_for_zip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("First, please upload the JSON file.", reply_markup=ReplyKeyboardRemove())
    return WAITING_JSON_TRANSLATE_ZIP

async def json_translate_get_json_for_zip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    json_doc = await update.message.document.get_file()
    json_bytes = await json_doc.download_as_bytearray()
    context.user_data['json_data'] = json.loads(json_bytes.decode('utf-8'))
    await update.message.reply_text("JSON received. Now, please upload the corresponding zip file.")
    return WAITING_ZIP_TRANSLATE

async def json_translate_process_zip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    zip_file = await update.message.document.get_file()
    json_data = context.user_data.pop('json_data')
    
    await update.message.reply_text("Processing zip file... This might take a while.")

    try:
        font = ImageFont.truetype(FONT_PATH, 20)
    except IOError:
        await update.message.reply_text(f"Error: Font file not found at '{FONT_PATH}'. Operation cancelled.")
        return await start(update, context)

    with tempfile.TemporaryDirectory() as temp_dir:
        input_zip_path = os.path.join(temp_dir, "input.zip")
        await zip_file.download_to_drive(input_zip_path)

        extract_path = os.path.join(temp_dir, "extracted")
        output_path = os.path.join(temp_dir, "output")
        os.makedirs(output_path)
        
        with zipfile.ZipFile(input_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

        for folder_info in json_data.get("folders", []):
            folder_name = folder_info["folder_name"]
            output_folder_path = os.path.join(output_path, folder_name)
            os.makedirs(output_folder_path, exist_ok=True)
            
            for image_info in folder_info.get("images", []):
                image_name = image_info["image_name"]
                image_path = os.path.join(extract_path, folder_name, image_name)
                
                if os.path.exists(image_path):
                    img = Image.open(image_path).convert("RGB")
                    draw = ImageDraw.Draw(img)
                    
                    for block in image_info.get("text_blocks", []):
                        x1, y1, x2, y2 = map(int, block["location"])
                        text = block["text"]
                        # Simple rectangle and text; can be fancier
                        draw.rectangle([x1, y1, x2, y2], fill="white")
                        draw.text((x1, y1), text, font=font, fill="black")
                    
                    img.save(os.path.join(output_folder_path, image_name))

        output_zip_path = os.path.join(temp_dir, "translated_comic.zip")
        shutil.make_archive(output_zip_path.replace('.zip', ''), 'zip', output_path)
        
        await update.message.reply_document(document=open(output_zip_path, 'rb'))

    await update.message.reply_text("Done! What would you like to do next? /start")
    return ConversationHandler.END

async def not_implemented(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("This feature is not yet implemented.", reply_markup=ReplyKeyboardRemove())
    return await start(update, context)


# --- 3. JSON Divide ---
async def json_divide_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    reply_keyboard = [["zip upload"]]
    await update.message.reply_text(
        "This function requires a master JSON file and a zip file.",
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True, resize_keyboard=True),
    )
    return JSON_DIVIDE_CHOICE

async def json_divide_prompt_json(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("First, please upload the master JSON file.", reply_markup=ReplyKeyboardRemove())
    return WAITING_JSON_DIVIDE

async def json_divide_get_json(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    json_doc = await update.message.document.get_file()
    json_bytes = await json_doc.download_as_bytearray()
    context.user_data['json_data'] = json.loads(json_bytes.decode('utf-8'))
    await update.message.reply_text("JSON received. Now, please upload the corresponding zip file.")
    return WAITING_ZIP_DIVIDE

async def json_divide_process_zip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    zip_file = await update.message.document.get_file()
    json_data = context.user_data.pop('json_data')

    await update.message.reply_text("Processing... This may take a while.")

    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, "input.zip")
        await zip_file.download_to_drive(zip_path)
        
        extract_path = os.path.join(temp_dir, "extracted")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

        for folder_info in json_data.get("folders", []):
            folder_name = folder_info["folder_name"]
            current_folder_path = os.path.join(extract_path, folder_name)
            
            if os.path.isdir(current_folder_path):
                folder_json_data = {"images": folder_info.get("images", [])}
                folder_json_path = os.path.join(current_folder_path, f"{folder_name}.json")
                with open(folder_json_path, 'w', encoding='utf-8') as f:
                    json.dump(folder_json_data, f, ensure_ascii=False, indent=4)
                
                for image_info in folder_info.get("images", []):
                    image_name = image_info["image_name"]
                    image_path = os.path.join(current_folder_path, image_name)
                    
                    if os.path.exists(image_path):
                        img = Image.open(image_path).convert("RGB")
                        draw = ImageDraw.Draw(img)
                        for block in image_info.get("text_blocks", []):
                            x1, y1, x2, y2 = map(int, block["location"])
                            draw.rectangle([x1, y1, x2, y2], fill="black")
                        img.save(image_path)
        
        output_zip_path = os.path.join(temp_dir, "divided_and_masked.zip")
        shutil.make_archive(output_zip_path.replace('.zip', ''), 'zip', extract_path)
        
        await update.message.reply_document(document=open(output_zip_path, 'rb'))
        
    await update.message.reply_text("Done! What would you like to do next? /start")
    return ConversationHandler.END


# --- 4. Choose OCR Language ---
async def choose_lang_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """This feature is disabled as Florence-2 is multilingual by default."""
    await update.message.reply_text(
        "The current OCR model (Florence-2) is multilingual by default and does not require language selection.",
        reply_markup=ReplyKeyboardRemove()
    )
    return await start(update, context)


# --- Media Group Handler ---
async def handle_media_group(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if 'media_group' not in context.user_data:
        context.user_data['media_group'] = []
    
    context.user_data['media_group'].append(update.message)

    job_name = f"process_media_group_{update.message.media_group_id}"
    # Schedule job only if it doesn't exist
    if not context.job_queue.get_jobs_by_name(job_name):
        context.job_queue.run_once(
            process_media_group_job, 
            when=2, # 2 second delay to collect all images
            data={'chat_id': update.effective_chat.id},
            name=job_name
        )

async def process_media_group_job(context: ContextTypes.DEFAULT_TYPE):
    """Job to process a collected media group."""
    class MockUpdate:
        def __init__(self, chat_id):
            self.effective_chat = type('chat', (), {'id': chat_id})
            self.message = type('message', (), {
                'reply_text': lambda *args, **kwargs: context.bot.send_message(chat_id, *args, **kwargs),
                'reply_document': lambda *args, **kwargs: context.bot.send_document(chat_id, *args, **kwargs)
            })()

    await json_maker_process_images(MockUpdate(context.job.data['chat_id']), context)

# --- Main Function to Run the Bot ---
def main() -> None:
    if BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("Error: Please replace 'YOUR_BOT_TOKEN_HERE' with your actual bot token.")
        return

    application = Application.builder().token(BOT_TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            MAIN_MENU: [
                MessageHandler(filters.Regex("^Json maker$"), json_maker_menu),
                MessageHandler(filters.Regex("^json To Comic translate$"), json_translate_menu),
                MessageHandler(filters.Regex("^json divide$"), json_divide_menu),
                MessageHandler(filters.Regex("^Choose ocr language$"), choose_lang_menu),
            ],
            JSON_MAKER_CHOICE: [
                MessageHandler(filters.Regex("^image upload$"), json_maker_prompt_image),
                MessageHandler(filters.Regex("^zip upload$"), json_maker_prompt_zip),
            ],
            WAITING_IMAGES_OCR: [
                MessageHandler(filters.PHOTO & filters.UpdateType.MESSAGE, handle_media_group),
            ],
            WAITING_ZIP_OCR: [MessageHandler(filters.Document.ZIP, json_maker_process_zip)],
            JSON_TRANSLATE_CHOICE: [
                MessageHandler(filters.Regex("^image upload$"), not_implemented),
                MessageHandler(filters.Regex("^zip upload$"), json_translate_prompt_json_for_zip),
            ],
            WAITING_JSON_TRANSLATE_ZIP: [MessageHandler(filters.Document.FileExtension('json'), json_translate_get_json_for_zip)],
            WAITING_ZIP_TRANSLATE: [MessageHandler(filters.Document.ZIP, json_translate_process_zip)],
            JSON_DIVIDE_CHOICE: [MessageHandler(filters.Regex("^zip upload$"), json_divide_prompt_json)],
            WAITING_JSON_DIVIDE: [MessageHandler(filters.Document.JSON, json_divide_get_json)],
            WAITING_JSON_DIVIDE: [MessageHandler(filters.Document.FileExtension('json'), json_divide_get_json)],
        },
        fallbacks=[CommandHandler("cancel", cancel), CommandHandler("start", start)],
    )

    application.add_handler(conv_handler)
    application.run_polling()


if __name__ == "__main__":
    main()
