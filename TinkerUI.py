from tkinter import *
import tkinter
from tkinter import filedialog
from Prediction import *

class Window(Frame):


    def __init__(self, master=None):
        Frame.__init__(self, master)                 
        self.master = master
        self.init_window()

    #Creation of init_window

    def browse_button(self):
        # Allow user to select a directory and store it in global var
        # called folder_path
        global folder_path
        filename = filedialog.askopenfilename()

        pd = Prediction()
        img_PIL, img_tensor = pd.get_img_tensor(filename)
        sentence = pd.get_caption(img_tensor)
        pd.visualize(img_PIL, sentence)


    def init_window(self):

        # changing the title of our master widget      
        self.master.title("GUI")

        # allowing the widget to take the full space of the root window
        self.pack(fill=BOTH, expand=1)

        # # creating a button instance
        Browse_Button = Button(self, text='Browse', command =self.browse_button)
        #
        # # placing the button on my window
        Browse_Button.place(relx=0.5, rely=0.33, anchor=CENTER)


        # # creating a button instance
        quitButton = Button(self, text="Quit")
        #
        # # placing the button on my window
        quitButton.place(relx=0.5, rely=0.66, anchor=CENTER)
        quitButton.bind("<Button-1>", self.button1Click)

    def button1Click(self, event):
        root.destroy()


root = Tk()
root.geometry("100x150")
windowWidth = root.winfo_reqwidth()
windowHeight = root.winfo_reqheight()
positionRight = int(root.winfo_screenwidth()/2 - windowWidth/2)
positionDown = int(root.winfo_screenheight()/2 - windowHeight/2)
root.geometry("+{}+{}".format(positionRight, positionDown))


app = Window(root)
root.mainloop()  