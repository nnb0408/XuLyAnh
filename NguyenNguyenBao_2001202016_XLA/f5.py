import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk,font
from PIL import ImageTk, Image

BLUE_COLOR = '#87CEFA'
class ImageSelector(tk.Frame):
    def __init__(self, master=None, background=BLUE_COLOR):
        super().__init__(master)
        self.master = master
        self.master.title("Thực Hành Xử lý Ảnh")
        self.configure(background=background)
        self.pack()
        self.button_frame = None

    def set_background_color(self, color):
        self.configure(background=color)
        self.button_frame.configure(background=color)
        self.canvas_frame.configure(background=color)
        self.label0.configure(background=BLUE_COLOR)
        self.label1.configure(background=BLUE_COLOR)
        self.label2.configure(background=BLUE_COLOR)
        self.label3.configure(background=BLUE_COLOR)

    def create_widgets(self):
        if self.button_frame is None:
            self.button_frame = tk.Frame(self)
        self.button_frame.pack(side="top", fill="x")
        # slt_ic = PhotoImage(file="C:\\bao\\Python\\NguyenNguyenBao_2001202016_XLA\\Images_Data\\select.png")
        self.label0 = tk.Label(self, text="Nguyễn Nguyên Bảo - 2001202016", font=("Arial", 28, "bold"), fg = 'red')#, background=BLUE_COLOR)
        # self.label0.configure(background=BLUE_COLOR)
        self.label0.pack(side="top", padx=15, pady=10)

        self.button_frame = tk.Frame(self)
        self.button_frame.pack(side="top", fill="x")
        self.select_button = tk.Button(self.button_frame, text="Chọn Ảnh", fg="white") #image=slt_ic, compound='left')#, command=self.select_image)
        self.select_button.pack(side="left", padx=15, pady=15)
        self.clear_button = tk.Button(self.button_frame, text="Xóa Ảnh", fg="white")#, command=self.clear_image)
        self.clear_button.pack(side="left", padx=35, pady=15)

        #Thiết lập màu cho cbb
        style = ttk.Style()
        style.theme_use('default')
        style.map('TCombobox', fieldbackground=[('readonly', 'white')])
        style.configure('TCombobox', foreground='black', background='black', selectbackground='green', height = 20)
        
        # Buttons
        self.select_button["font"] = ("Arial", 20, "bold")
        self.select_button["bg"] = "green"
        self.clear_button["font"] = ("Arial", 20, "bold")
        self.clear_button["bg"] = "red"
        
        # Combobox thứ nhất
        self.combo1 = ttk.Combobox(self.button_frame, state="readonly", values=['K-Mean', 'Phát Hiện Đường Viền', 'Ngưỡng Otsu', 'Dùng Mặt Nạ Màu'], style='TCombobox')
        self.combo1.set('Phân Đoạn Ảnh')
        self.combo1.pack(side="left", padx=25, pady=15)

        # Combobox thứ hai
        self.combo2 = ttk.Combobox(self.button_frame, state="readonly", values=['Option A', 'Option B', 'Option C'])
        self.combo2.set('Phân Ngưỡng')
        self.combo2.pack(side="left", padx=25, pady=15)

        # Combobox thứ ba
        self.combo3 = ttk.Combobox(self.button_frame, state="readonly", values=['Choice 1', 'Choice 2', 'Choice 3'])
        self.combo3.set('Nâng Cao Chất Lượng')
        self.combo3.pack(side="left", padx=25, pady=15)

        # Combobox thứ tư
        self.combo4 = ttk.Combobox(self.button_frame, state="readonly", values=['Phép Co','Phép Giãn','Phép Mở','Phép Đóng','Morphological Gradient','Top Hat','Black Hat','Hit Miss', 'Rectange', 'Cross'])
        self.combo4.set('Xử Lý Hình Thái')
        self.combo4.pack(side="left", padx=25, pady=15)
        
        # Comboboxes
        self.combo1["font"] = ("Arial", 15) 
        self.combo2["font"] = ("Arial", 15) 
        self.combo3["font"] = ("Arial", 15) 
        self.combo4["font"] = ("Arial", 15)

        # Canvas thứ nhất chứa ảnh gốc
        self.canvas_frame = tk.Frame(self)
        self.canvas_frame.pack(side="left", fill="both", expand=True, padx=15)
    
        self.canvas = tk.Canvas(self.canvas_frame, width=480, height=480, background="light gray")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.label1 = tk.Label(self, text="Ảnh Ban Đầu", font=("Arial", 15, "bold"), foreground="green")
        self.label1.place(x=330,y=180)

        self.label3 = tk.Label(self.canvas_frame, text="➜", font=("Arial", 15, "bold"))
        self.label3.pack(side="left", padx=10, pady=10)

        # Canvas thứ hai chứa ảnh biến đổi
        self.canvas2 = tk.Canvas(self.canvas_frame, width=480, height=480, background="light gray")
        self.canvas2.pack(side="left", fill="both", expand=True)

        self.label2 = tk.Label(self, text="Ảnh Đã Chỉnh Sửa", font=("Arial", 15, "bold"), foreground="green")
        self.label2.place(x=1030,y=180)

        self.canvas_frame.pack(side="left", fill="both", expand=True, padx=15, pady=70)


        self.image_path = None
        self.image = None
        self.photo = None

        self.set_background_color(BLUE_COLOR)

        style.map('TCombobox', foreground=[('readonly','black'), ('selected', 'red')])

        self.combo1.bind("<<ComboboxSelected>>", self.update_image_combobox1)



    # def on_select_combo1(self, index):
    #     if index >= 0:
    #         # Thiết lập màu nền của Combobox bị mất 
    #         self.combo1.configure(background='')
            
    #         # # Thiết lập màu chữ của Combobox thành màu đỏ
    #         # style = ttk.Style()
    #         # style.map('TCombobox', fieldbackground=[('readonly','white')])
    #         # style.map('TCombobox', foreground=[('readonly','black'), ('selected', 'red')])
    #         # style.map('TCombobox', selectbackground=[('readonly','white'), ('selected', 'white')])
    #         # style.configure('TCombobox', selectforeground='red')
    
    def clear_image(self):
        self.canvas.delete("all")
        self.canvas2.delete("all")
        self.image_path = None
        self.image = None
        self.photo = None

    def update_image_combobox1(self, event=None):
        if self.image_path is None:
            return

        try:
            index = self.combo1.current() + 1
            # K_mean
            if index == 1:
                # self.on_select_combo1(index)
                index

            # Phân đoạn hình ảnh bằng phát hiện đường viền
            if index == 2:
                # self.on_select_combo1(index)
                index

            # Dùng ngưỡng Otsu
            if index == 3:
                # self.on_select_combo1(index)
                index

            # Dùng mặt nạ màu:
            if index == 4:
                # self.on_select_combo1(index)
                index

            self.cout += 1
            # TODO: Cập nhật ảnh mới tại đây
        except:
            messagebox.showerror("Có lỗi!", "Không thể mở hình ảnh.")
        
        self.on_select_combo1(self.combo1.current())
    
root = tk.Tk()
app = ImageSelector(master=root, background=BLUE_COLOR)
app.create_widgets()
app.mainloop()

