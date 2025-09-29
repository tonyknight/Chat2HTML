import os
from dotenv import load_dotenv

# Import the new classes
from src.database_manager import DatabaseManager
from src.html_generator import HTMLGenerator
from src.parsers.whatsapp_parser import WhatsAppParser
from src.parsers.messenger_parser import MessengerParser
from src.parsers.imessage_parser import IMessageParser

def main_menu():
    """Displays the main menu and handles user choices for parsing."""
    print("\n--- Chat Archiver ---")
    print("A unified tool to parse and archive chat logs.")
    print("\nSelect an option:")
    print("  1. Import WhatsApp Chats")
    print("  2. Import Facebook Messenger Chats")
    print("  3. Process iMessage Export")
    print("  4. Generate HTML Reports")
    print("  5. Exit")

    choice = input("Enter your choice: ").strip()
    return choice

def process_and_import(parser, service_name, db_manager):
    """Handles parsing a source and inserting it into the database."""
    source_path = input(f"Enter the path to your {service_name} data: ").strip()
    print(f"Parsing {service_name} data...")
    messages = parser.parse(source_path)
    if messages:
        print(f"Found {len(messages)} messages. Inserting into database...")
        with db_manager:
            db_manager.insert_messages(messages, service_name)
        print("Insertion complete.")
    else:
        print("No messages found or an error occurred during parsing.")

def generate_html_submenu(db_manager):
    """Handles the generation of HTML reports from the database."""
    html_generator = HTMLGenerator()
    # Placeholder for a more advanced submenu
    print("\nGenerating 'All Chat' report...")
    with db_manager:
        all_messages = db_manager.get_all_messages()
    
    if all_messages:
        owner_name = input("Enter the archive owner's name (for 'sent' messages): ").strip()
        output_path = os.path.join("output", "All_Chats_Archive.html")
        html_generator.generate_html(all_messages, output_path, main_author=owner_name)
    else:
        print("No messages found in the database to generate a report.")

def main():
    """Main function to orchestrate the chat export processing."""
    load_dotenv("API_Key.env")
    os.makedirs("output", exist_ok=True)
    
    db_path = os.path.join("output", "chat_archive.db")
    db_manager = DatabaseManager(db_path=db_path)

    # Ensure tables are created once at the start
    with db_manager:
        db_manager.create_tables()

    while True:
        choice = main_menu()
        
        if choice == '1':
            process_and_import(WhatsAppParser(), "WhatsApp", db_manager)
        elif choice == '2':
            process_and_import(MessengerParser(), "Facebook Messenger", db_manager)
        elif choice == '3':
            # iMessage is a special case that handles its own DB interaction for now
            # In a future refactoring, this could also be standardized.
            parser = IMessageParser()
            imessage_data = parser.parse(input("Enter path to iMessage export directory: ").strip())
            # We would then pass this to a specialized DB import function.
            # For now, we assume the parser handles it.
        elif choice == '4':
            generate_html_submenu(db_manager)
        elif choice == '5':
            print("Exiting.")
            break
        else:
            print("Invalid choice. Please try again.")
        
        print("\n--------------------")


if __name__ == "__main__":
    main() 
