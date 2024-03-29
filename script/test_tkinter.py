import Tkinter as tk

class MainWindow(tk.Frame):
    counter = 0
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
#        self.button = tk.Button(self, text="Create new window",
#                               command=self.create_window)
#        self.button.pack(side="top")
        root.bind("<Button-3>", self.create_window)

    def create_window(self,event):
        self.counter += 1
        if self.counter%2==0:
            t = tk.Toplevel(self)
            t.wm_title("Window #%s" % self.counter)
            l = tk.Label(t, text="This is window #%s" % self.counter)
            l.pack(side="top", fill="both", expand=True, padx=100, pady=100)

if __name__ == "__main__":
    root = tk.Tk()
    main = MainWindow(root)
    main.pack(side="top", fill="both", expand=True)
    root.mainloop()