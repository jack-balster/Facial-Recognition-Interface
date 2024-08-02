import sqlite3

def delete_all_names():
    conn = sqlite3.connect('face_recognition.db')  # Adjust the path if your database file is in a different location
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users")  # Assuming your table name is 'users'
    conn.commit()
    conn.close()
    print("All names have been deleted from the database.")

if __name__ == "__main__":
    delete_all_names()
