from face_recognition import face_recognition
from create_classifier import train_classifer
from create_dataset import start_capture
from gender_prediction import age_and_gender_prediction, emotion_prediction
import tkinter as tk
import sqlite3
from tkinter import ttk
from tkinter import font as tkfont
from tkinter import messagebox, PhotoImage

# Global set to store names
names = set()

# Load names from the database into a set
def load_names():
    global names
    conn = sqlite3.connect('face_recognition.db')
    c = conn.cursor()
    c.execute("SELECT name FROM users")
    rows = c.fetchall()
    names = {row[0] for row in rows}
    conn.close()

# Save a new name to the database
def save_name(name):
    conn = sqlite3.connect('face_recognition.db')
    c = conn.cursor()
    c.execute("INSERT INTO users (name) VALUES (?)", (name,))
    conn.commit()
    conn.close()

# Function to save names from the global set to a file
def save_names():
    with open("nameslist.txt", "w") as f:
        f.write(" ".join(names))

# Main UI class
class MainUI(tk.Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        load_names()  # Load names on startup
        self.title_font = tkfont.Font(family='Helvetica', size=16, weight="bold")
        self.title("Facial Recognition App")
        self.resizable(False, False)
        self.geometry("550x300")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.active_name = None  # Currently active user name
        self.num_of_images = 0  # Number of images captured

        # Container for frames
        container = tk.Frame(self)
        container.grid(sticky="nsew")
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # Dictionary to store frames
        self.frames = {}
        for F in (StartPage, PageOne, PageTwo, PageThree, PageFour):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame("StartPage")  # Show the start page initially

    # Function to show a specific frame
    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()

    # Function to handle the closing of the application
    def on_closing(self):
        if messagebox.askokcancel("Quit", "Are you sure?"):
            save_names()
            self.destroy()

# Start page class
class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        render = PhotoImage(file='assets/homepagepic.png')
        img = tk.Label(self, image=render)
        img.image = render
        img.grid(row=0, column=3, rowspan=4, padx=55, pady=15, sticky="nsew")  # Move to column 3 and add padding

        label = tk.Label(self, text="Menu", font=self.controller.title_font, bg="#263942", fg="#ffffff")
        label.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        button1 = ttk.Button(self, text="Sign up", command=lambda: self.controller.show_frame("PageOne"))
        button2 = ttk.Button(self, text="Check a User", command=lambda: self.controller.show_frame("PageTwo"))
        button3 = ttk.Button(self, text="Quit", command=self.on_closing)
        button1.grid(row=1, column=0, padx=20, pady=1, sticky="ew")
        button2.grid(row=2, column=0, padx=20, pady=1, sticky="ew")
        button3.grid(row=3, column=0, padx=20, pady=1, sticky="ew")

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Are you sure?"):
            save_names()
            self.controller.destroy()

# Signup page (enter username and navigate to 2nd signup page)
class PageOne(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        tk.Label(self, text="Enter a username", bg="#ffffff", fg="#263942", font='Helvetica 12').grid(row=0, column=0, pady=10, padx=5)
        self.user_name = ttk.Entry(self, font='Helvetica 11')
        self.user_name.grid(row=0, column=1, pady=10, padx=10)
        self.buttonback = ttk.Button(self, text="Back", command=lambda: controller.show_frame("StartPage"))
        self.buttonext = ttk.Button(self, text="Next", command=self.start_training)
        self.buttonback.grid(row=1, column=0, padx=10, pady=20, ipadx=5, ipady=4)
        self.buttonext.grid(row=1, column=1, pady=10, ipadx=5, ipady=4)

    # Function to handle the start of the training process
    def start_training(self):
        global names
        name = self.user_name.get().strip()
        if name == "None":
            messagebox.showerror("Error", "Name cannot be 'None'")
            return
        elif name in names:
            messagebox.showerror("Error", "User already exists!")
            return
        elif not name:
            messagebox.showerror("Error", "Name cannot be empty!")
            return
        names.add(name)
        save_name(name)
        self.controller.active_name = name
        self.controller.frames["PageTwo"].refresh_names()
        self.controller.show_frame("PageThree")

    def clear(self):
        self.user_name.delete(0, 'end')

# Login and navigate to functionality page by selecting an existing user
class PageTwo(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        tk.Label(self, text="Select your username", bg="#ffffff", fg="#263942", font='Helvetica 12').grid(row=0, column=0, padx=10, pady=10)

        self.menuvar = tk.StringVar(self)
        self.menuvar.set('Select a user')
        style = ttk.Style()
        style.configure('TMenubutton', background='#ffffff', foreground='#263942', font='Helvetica 11')
        self.dropdown = ttk.OptionMenu(self, self.menuvar, 'Select a user', *names, command=self.update_active_name)
        self.dropdown.grid(row=0, column=1, padx=10, pady=10)
        self.buttonext = ttk.Button(self, text="Next", command=self.next_foo)
        self.buttonext.grid(row=1, column=1, pady=10, ipadx=5, ipady=4)
        self.buttonback = ttk.Button(self, text="Back", command=lambda: controller.show_frame("StartPage"))
        self.buttonback.grid(row=1, column=0, pady=10, ipadx=5, ipady=4)

    # Function to update the active user name based on the selection
    def update_active_name(self, selection):
        self.controller.active_name = selection
        print(f"Selected user: {self.controller.active_name}")

    # Function to handle navigation to the next page
    def next_foo(self):
        if self.menuvar.get() == 'Select a user':
            messagebox.showerror("ERROR", "No user selected. Please select a user.")
            return
        self.controller.show_frame("PageFour")
        
    def refresh_names(self):
        global names
        self.dropdown['menu'].delete(0, "end")
        for name in names:
            self.dropdown['menu'].add_command(label=name, command=lambda value=name: self.menuvar.set(value))

# 2nd Signup page capturing images and training the model
class PageThree(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.numimglabel = tk.Label(self, text="Number of images captured = 0", font='Helvetica 12', bg="#ffffff", fg="#263942")
        self.numimglabel.grid(row=0, column=0, columnspan=2, sticky="ew", pady=10)
        self.capturebutton = ttk.Button(self, text="1. Capture Images", command=self.capimg)
        self.trainbutton = ttk.Button(self, text="2. Train The Model", command=self.trainmodel)
        self.buttoncanc = ttk.Button(self, text="Back", command=lambda: controller.show_frame("PageOne"))
        self.capturebutton.grid(row=1, column=0, ipadx=5, ipady=4, padx=10, pady=20)
        self.trainbutton.grid(row=1, column=1, ipadx=5, ipady=4, padx=10, pady=20)
        self.buttoncanc.grid(row=2, column=0, ipadx=5, ipady=4, padx=10, pady=20)

    # Function to capture images
    def capimg(self):
        self.numimglabel.config(text="Captured Images = 0")
        messagebox.showinfo("INSTRUCTIONS", "We will start capturing 300 images: ")
        x = start_capture(self.controller.active_name)
        self.controller.num_of_images = x
        self.update_num_of_images(x)
        self.numimglabel.config(text=f"Number of images captured = {x}")

    def update_num_of_images(self, num_of_images):
        conn = sqlite3.connect('face_recognition.db')
        c = conn.cursor()
        c.execute("UPDATE users SET num_of_images = ? WHERE name = ?", (num_of_images, self.controller.active_name))
        conn.commit()
        conn.close()

    # Function to train the model
    def trainmodel(self):
        if self.controller.num_of_images < 300:
            messagebox.showerror("ERROR", "Not enough data, capture at least 300 images first!")
            return
        train_classifer(self.controller.active_name)
        messagebox.showinfo("SUCCESS", "The model has been successfully trained!")
        self.controller.show_frame("PageFour")

# Logged in functionality (emotion, age, gender prediction)
class PageFour(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        label = tk.Label(self, text="Program Options", font='Helvetica 16 bold')
        label.grid(row=0,column=0, sticky="ew")
        button1 = ttk.Button(self, text="Face Recognition", command=self.faceRecognition)
        button2 = ttk.Button(self, text="Emotion Detection", command=self.emotionPrediction)
        button3 = ttk.Button(self, text="Gender and Age Prediction", command=self.gender_age_pred)
        button5 = ttk.Button(self, text="Back", command=lambda: controller.show_frame("PageTwo"))
        button4 = ttk.Button(self, text="Go to Home Page", command=lambda: self.controller.show_frame("StartPage"))
        button1.grid(row=1,column=0, sticky="ew", ipadx=5, ipady=4, padx=10, pady=10)
        button2.grid(row=2,column=0, sticky="ew", ipadx=5, ipady=4, padx=10, pady=10)
        button3.grid(row=3,column=0, sticky="ew", ipadx=5, ipady=4, padx=10, pady=10)
        button5.grid(row=4, column=0, ipadx=5, ipady=4, padx=10, pady=20)
        button4.grid(row=4,column=1, sticky="ew", ipadx=5, ipady=4, padx=10, pady=10)

    # Function to open the webcam for face recognition
    def faceRecognition(self):
        face_recognition(self.controller.active_name)
        
    # Function to perform gender and age prediction
    def gender_age_pred(self):
        age_and_gender_prediction()

    # Function to perform emotion prediction
    def emotionPrediction(self):
        emotion_prediction()

app = MainUI()
app.iconphoto(True, tk.PhotoImage(file='assets/icon.ico'))
app.mainloop()
