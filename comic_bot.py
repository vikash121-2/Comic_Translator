async def json_translate_get_json_for_img(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    json_content = None

    # Case 1: User uploaded a JSON file directly
    if update.message.document and update.message.document.file_name.endswith('.json'):
        json_file = await update.message.document.get_file()
        json_content = await json_file.download_as_bytearray()

    # Case 2: User sent a text message with a URL
    elif update.message.text:
        url = update.message.text
        if url.startswith("http") and url.endswith(".json"):
            await update.message.reply_text("URL detected. Downloading JSON...")
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(url)
                    response.raise_for_status()  # Raise an exception for bad status codes
                    json_content = response.content
            except httpx.RequestError as e:
                await update.message.reply_text(f"Failed to download from URL: {e}")
                return WAITING_JSON_TRANSLATE_IMG # Ask again
        else:
            await update.message.reply_text("This doesn't look like a valid JSON URL. Please send a file or a valid link.")
            return WAITING_JSON_TRANSLATE_IMG # Ask again

    if json_content:
        try:
            context.user_data['json_data'] = json.loads(json_content)
            context.user_data['temp_dir_obj'] = tempfile.TemporaryDirectory(dir=TEMP_ROOT_DIR)
            context.user_data['received_images'] = {}
            await update.message.reply_text("JSON received. Now send the original images. Press 'Done' when finished.")
            return WAITING_IMAGES_TRANSLATE
        except json.JSONDecodeError:
            await update.message.reply_text("The file or URL content is not valid JSON. Please try again.")
            return WAITING_JSON_TRANSLATE_IMG
    
    # If neither a valid file nor a valid URL was provided
    return WAITING_JSON_TRANSLATE_IMG
