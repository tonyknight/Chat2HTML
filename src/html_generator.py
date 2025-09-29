from jinja2 import Environment, FileSystemLoader
import os
import datetime

class HTMLGenerator:
    """Handles the generation of the final HTML report."""
    def __init__(self, template_dir="templates"):
        self.env = Environment(loader=FileSystemLoader(template_dir))
        self.template = self.env.get_template("template.html")

    def generate_html(self, messages, output_path, main_author=None):
        """
        Renders the HTML file from a list of processed message dictionaries.
        
        Args:
            messages (list): The list of message dictionaries.
            output_path (str): The full path where the HTML file will be saved.
            main_author (str, optional): The name of the main user/owner. Defaults to None.
        """
        if not messages:
            print("No messages to generate HTML for.")
            return

        # Simplified data preparation for now. Will be expanded.
        # This part will need to be made more robust to handle the full calendar, etc.
        messages_by_date = {}
        for msg in messages:
            # Ensure timestamp is a datetime object
            if isinstance(msg['timestamp'], str):
                msg['timestamp'] = datetime.datetime.fromisoformat(msg['timestamp'])

            date = msg["timestamp"].date()
            if date not in messages_by_date:
                messages_by_date[date] = []
            messages_by_date[date].append(msg)
            
        # For now, we'll pass an empty calendar and a placeholder main_author
        # This will be replaced with a more robust data preparation step.
        calendar_data = [] # Placeholder
        if not main_author and messages:
            # A simple heuristic to find the main author for non-iMessage chats
             try:
                all_authors = list(dict.fromkeys([m["message_sender"] for m in messages]))
                main_author = all_authors[1] if len(all_authors) > 1 else all_authors[0]
             except:
                 main_author = "Me"


        html_content = self.template.render(
            messages_by_date=messages_by_date,
            main_author=main_author,
            calendar_data_by_year=calendar_data # Placeholder
        )
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        print(f"HTML file generated at: {output_path}")
