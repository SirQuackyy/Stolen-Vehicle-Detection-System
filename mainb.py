import customtkinter
import os
customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("dark-blue")


hub = customtkinter.CTk()
hub.title("LICENSE TO STEAL")


def uploadVideo():
    path = customtkinter.filedialog.askopenfile("*.mp4")
    print(path)

def switchToLive():
    os.system("main.py")


button1 = customtkinter.CTkButton(hub, text="Live Feed",command=switchToLive)
button1.pack()

button2 = customtkinter.CTkButton(hub, text="Upload File", command=uploadVideo)
button2.pack()


hub.mainloop()