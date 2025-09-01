# WhatsApp Chat HTML Exporter

This Python script converts a WhatsApp chat text export into a beautiful, self-contained single-page HTML file. It includes features like automatic translation, image processing, and an interactive, modern UI for easy viewing of your chat history.

## Features

- **Interactive Chat UI**: Displays your chat log in a clean, modern interface with distinct bubbles for each author, styled in the beautiful Nord dark theme.
- **Automatic Translation**: Automatically detects non-English messages and translates them to English using a configurable OpenRouter AI model. The original text is preserved alongside the translation for easy comparison.
- **Smart Image Handling**:
    - **EXIF Tagging**: Intelligently parses timestamps from image filenames and uses `exiftool` to write them to the `DateTimeOriginal` EXIF tag.
    - **File Renaming**: Renames image files based on their timestamp for clear, chronological organization.
    - **Image Galleries**: Automatically groups sequential images from the same author within a 5-minute window into a compact, beautiful gallery.
- **Interactive Calendar**: Generates a calendar header that spans the duration of your chat. Days with messages are highlighted and clickable, allowing you to jump directly to the first message of that day.
- **Image Lightbox**: Click on any image thumbnail to open a full-screen lightbox preview. You can easily navigate between all images in the chat using the on-screen arrows or your keyboard's left and right arrow keys.
- **Efficient & Parallel Processing**: Uses a multi-threaded approach to process images and perform translations in parallel, significantly speeding up the export process for large chat logs.
- **Error Logging**: Logs any issues encountered during the process (e.g., missing images, failed translations) to a timestamped text file for easy debugging.

## Prerequisites

- [Python 3.6+](https://www.python.org/downloads/)
- [ExifTool](https://exiftool.org/install.html) must be installed and accessible in your system's PATH.

## Setup & Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/tonyknight/WAChat2HTML.git
    cd WAChat2HTML
    ```

2.  **Create a Virtual Environment**: It is highly recommended to use a virtual environment to manage dependencies.
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
    *(On Windows, the activate command is `venv\Scripts\activate`)*

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure the Script**: Open the `main.py` script and edit the configuration section at the top:
    - `CHAT_FILE_PATH`: Set the absolute path to your exported `_chat.txt` file.
    - `PERFORM_TRANSLATION`: Set to `True` to enable automatic translation.
    - `MAX_WORKERS`: Adjust the number of parallel workers for processing. A good starting point is 10-25.

5.  **Set Up API Key (for Translation)**: If you enable translation, you must provide your OpenRouter API key. It is recommended to set this as an environment variable for security.

    - **macOS/Linux**:
      ```bash
      export OPENROUTER_API_KEY="your_api_key_here"
      ```
    - **Windows**:
      ```bash
      set OPENROUTER_API_KEY="your_api_key_here"
      ```
    Alternatively, you can hardcode the key in the script, but this is not recommended for public repositories.

## How to Run

Once the setup is complete, simply run the script from your terminal:

```bash
python3 main.py
```

The script will process the chat file and generate the output in the `/output` directory. You can open the `index.html` file in any modern web browser to view your chat log. A summary report, including processing stats and any errors, will be printed to the console upon completion.

## Project Structure

```
/
|-- main.py                 # The main Python script
|-- requirements.txt        # Python package dependencies
|-- templates/
|   |-- template.html       # Jinja2 HTML template for the chat log
|-- output/                 # Default directory for generated files
    |-- index.html          # The final, self-contained HTML chat log
    |-- images/             # Folder for all processed images
```

