import sqlite3

def initialize_db():
    conn = sqlite3.connect('face_recognition.db')
    c = conn.cursor()
    
    # Create users table
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    num_of_images INTEGER DEFAULT 0
                 )''')
    conn.commit()
    conn.close()

if __name__ == '__main__':
    initialize_db()
