import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from tkinter import Canvas
# from PIL import ImageTk, Image
# import cv2, pathlib, os
# import numpy as np

class ImageSelector(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("Đồ án Thực hành Xử lý Ảnh")
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        # self.image = Image.open("animal_background.png") 
        # self.photo = ImageTk.PhotoImage(self.image)
        # self.label_bg = tk.Label(self, image = self.photo)
        # self.label_bg.place(x=0, y=0, relwidth=1, relheight=1) 

        self.label0 = tk.Label(self, text="Chọn và Xử lý Ảnh - Nguyễn Nguyên Bảo - 2001202016", font=("Arial", 28), fg = 'green')
        self.label0.pack(side="top", padx=15, pady=10)

        self.button_frame = tk.Frame(self)
        self.button_frame.pack(side="top", fill="x")
        self.select_button = tk.Button(self.button_frame, text="Chọn Ảnh")#, command=self.select_image)
        self.select_button.pack(side="left", padx=15, pady=15)
        self.clear_button = tk.Button(self.button_frame, text="Xóa Ảnh")#, command=self.clear_image)
        self.clear_button.pack(side="left", padx=35, pady=15)
        
        # Buttons
        self.select_button["font"] = ("Arial", 20)
        self.select_button["bg"] = "green"
        self.clear_button["font"] = ("Arial", 20)
        self.clear_button["bg"] = "red"
        
        # Combobox thứ nhất
        self.combo1 = ttk.Combobox(self.button_frame, state="readonly", values=['K-Mean', 'Phát Hiện Đường Viền', 'Ngưỡng Otsu', 'Dùng Mặt Nạ Màu'])
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
        
        self.label0 = tk.Label(self, text=" ")
        self.label0.pack(side="top", padx=15, pady=5)
        self.label1 = tk.Label(self, text="Ảnh Gốc", font=("Arial", 15, "bold"), foreground="green")
        self.label1.place(x=350,y=160)
        self.label2 = tk.Label(self, text="Ảnh đã Biến đổi", font=("Arial", 15, "bold"), foreground="green")
        self.label2.place(x=1100,y=160)

        # Comboboxes
        self.combo1["font"] = ("Arial", 15) 
        self.combo2["font"] = ("Arial", 15) 
        self.combo3["font"] = ("Arial", 15) 
        self.combo4["font"] = ("Arial", 15) 

        self.combo1['bg'] = 'blue'         
        self.combo1['activeforeground'] = 'white'

        self.combo2['bg'] = 'green'       
        self.combo2['activeforeground'] = 'yellow'

        self.combo3['bg'] = 'red'       
        self.combo3['activeforeground'] = 'white' 

        self.combo4['bg'] = 'orange'          
        self.combo4['activeforeground'] = 'black'

        # Canvas thứ nhất chứa ảnh gốc
        self.canvas = tk.Canvas(self, width=500, height=500, background="light gray")
        self.canvas.pack(side="left", fill="both", expand=True, padx=15)

        # Canvas thứ hai chứa ảnh biến đổi
        self.canvas2 = tk.Canvas(self, width=500, height=500, background="light gray")
        self.canvas2.pack(side="left", fill="both", padx=15, expand=True)


        self.image_path = None
        self.image = None
        self.photo = None

        # # đăng ký sự kiện change của các combobox
        # self.combo1.bind("<<ComboboxSelected>>", self.update_image_combobox1)
        # self.combo2.bind("<<ComboboxSelected>>", self.update_image_combobox2)
        # self.combo3.bind("<<ComboboxSelected>>", self.update_image_combobox3)
        # self.combo4.bind("<<ComboboxSelected>>", self.update_image_combobox4)

    # def update_image_combobox1(self, event=None):
    #     if self.image_path is None:
    #         return

    #     # try:
    #     # Tạo thư mục lưu kết quả
    #     p = pathlib.Path('./KQ_PDA')
    #     p.mkdir(exist_ok=True)
    #     file_path = os.path.abspath('./KQ_PDA')
    #     print(file_path)
    #     # Đọc ảnh vào
    #     img = cv2.imread(self.image_path)
        
    #     self.combo2.set('Phân Ngưỡng')
    #     self.combo3.set('Nâng Cao Chất Lượng')
    #     self.combo4.set('Xử Lý Hình Thái')

    #     index = self.combo1.current() + 1
        
    #     if index == 1:
    #         # Sử Dụng K-Mean
    #         twoDimage = img.reshape((-1,3))
    #         twoDimage = np.float32(twoDimage)
    #         print(twoDimage)

    #         criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    #         K = 4
    #         attempts = 10

    #         label = cv2.kmeans(twoDimage, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    #         center = cv2.kmeans(twoDimage, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    #         center = np.uint8(center)
    #         res = center[label.flatten()]
    #         k_mean = res.reshape((img.shape))

    #         # img = cv2.resize(img, (500, 500))
    #         cv2.imwrite(file_path + "\k_mean.jpg", k_mean)                
    #         self.image2 = Image.open(file_path + "\k_mean.jpg")
            
    #     self.photo2 = ImageTk.PhotoImage(self.image2.resize((500, 500), Image.Resampling.LANCZOS))
    #     self.canvas2.create_image(0, 0, anchor='nw', image=self.photo2)   
    #     # except:
    #     #     messagebox.showerror("Có lỗi!", "Không thể mở hình ảnh.")
    
    # def update_image_combobox2(self, event=None):
    #     if self.image_path is None:
    #         return
    #     try:
    #         # Tạo thư mục lưu kết quả
    #         p = pathlib.Path('./KQ_PN')
    #         p.mkdir(exist_ok=True)
    #         file_path = os.path.abspath('./KQ_PN')

    #         index = self.combo2.current() + 1
    #         if index == 1:
    #             self.combo1.set('Phân Đoạn Ảnh')
    #             self.combo3.set('Nâng Cao Chất Lượng')
    #             self.combo4.set('Xử Lý Hình Thái')
    #             self.image2 = Image.open("D:\BTVN\TH_XLA\Images_Data\img02.png")
    #             self.photo2 = ImageTk.PhotoImage(self.image2.resize((500, 500), Image.Resampling.LANCZOS))
    #             self.canvas2.create_image(0, 0, anchor='nw', image=self.photo2)
    #     except:
    #         messagebox.showerror("Có lỗi!", "Không thể mở hình ảnh.")
    
    # def update_image_combobox3(self, event=None):
    #     if self.image_path is None:
    #         return
    #     try:
    #         # Tạo thư mục lưu kết quả
    #         p = pathlib.Path('./KQ_NCCL')
    #         p.mkdir(exist_ok=True)
    #         file_path = os.path.abspath('./KQ_NCCL')

    #         index = self.combo3.current() + 1

    #         if index == 1:
    #             self.combo2.set('Phân Ngưỡng')
    #             self.combo1.set('Phân Đoạn Ảnh')
    #             self.combo4.set('Xử Lý Hình Thái')
    #             self.image2 = Image.open("D:\BTVN\TH_XLA\Images_Data\img02.png")
    #             self.photo2 = ImageTk.PhotoImage(self.image2.resize((500, 500), Image.Resampling.LANCZOS))
    #             self.canvas2.create_image(0, 0, anchor='nw', image=self.photo2)
    #     except:
    #         messagebox.showerror("Có lỗi!", "Không thể mở hình ảnh.")
    
    # def update_image_combobox4(self, event=None):
    #     if self.image_path is None:
    #         return
    #     try:
    #         # Tạo thư mục lưu kết quả
    #         p = pathlib.Path('./KQ_XLHT')
    #         p.mkdir(exist_ok=True)
    #         file_path = os.path.abspath('./KQ_XLHT')

    #         # Đọc ảnh vào
    #         self.img = cv2.imread(self.image_path,0)
    #         # Tạo ma trận có kích thước 10x10
    #         self.kernel = np.ones((10,10), np.uint8)
    #         # Áp dụng các phép với kernel đã tạo 

    #         self.combo2.set('Phân Ngưỡng')
    #         self.combo3.set('Nâng Cao Chất Lượng')
    #         self.combo1.set('Phân Đoạn Ảnh')

    #         index = self.combo4.current() + 1

    #         if index == 1:
    #             # Phép co:
    #             self.erosion = cv2.erode(self.img, self.kernel, iterations = 1)
    #             self.erosion = cv2.resize(self.erosion, (500, 500))
    #             cv2.imwrite(file_path + "\erosion.jpg", self.erosion)                
    #             self.image2 = Image.open(file_path + "\erosion.jpg")

    #         elif index == 2:
    #             # Phép giãn:
    #             self.dilation = cv2.dilate(self.img, self.kernel, iterations = 1)
    #             self.dilation = cv2.resize(self.dilation, (500, 500))
    #             cv2.imwrite(file_path + "\dilation.jpg", self.dilation)
    #             self.image2 = Image.open(file_path + "\dilation.jpg")
            
    #         elif index == 3:
    #             # Phép mở:
    #             self.opening = cv2.morphologyEx(self.img, cv2.MORPH_OPEN, self.kernel)
    #             self.opening = cv2.resize(self.opening, (500, 500))
    #             cv2.imwrite(file_path + "\opening.jpg", self.opening)
    #             self.image2 = Image.open(file_path + "\opening.jpg")

            
    #         elif index == 4:
    #             # Phép đóng:
    #             self.closing = cv2.morphologyEx(self.img, cv2.MORPH_CLOSE, self.kernel)
    #             self.closing = cv2.resize(self.closing, (500, 500))
    #             cv2.imwrite(file_path + "\closing.jpg", self.closing)
    #             self.image2 = Image.open(file_path + "\closing.jpg")

    #         elif index == 5:
    #             # Morphological Gradient
    #             self.gradient = cv2.morphologyEx(self.img, cv2.MORPH_GRADIENT, self.kernel)
    #             self.gradient = cv2.resize(self.gradient, (500, 500))
    #             cv2.imwrite(file_path + "\gradient.jpg", self.gradient)
    #             self.image2 = Image.open(file_path + "\gradient.jpg")

    #         elif index == 6:
    #             # Toán tử Top Hat
    #             self.tophat = cv2.morphologyEx(self.img, cv2.MORPH_TOPHAT, self.kernel)
    #             self.tophat = cv2.resize(self.tophat, (500, 500))
    #             cv2.imwrite(file_path + "\_tophat.jpg", self.tophat)
    #             self.image2 = Image.open(file_path + "\_tophat.jpg")

    #         elif index == 7:
    #             # Toán tử Black Hat
    #             self.blackhat = cv2.morphologyEx(self.img, cv2.MORPH_BLACKHAT, self.kernel)
    #             self.blackhat = cv2.resize(self.blackhat, (500, 500))
    #             cv2.imwrite(file_path + "\_blackhat.jpg", self.blackhat)
    #             self.image2 = Image.open(file_path + "\_blackhat.jpg")
                
    #         elif index == 8:
    #             # Toán tử Hit Miss
    #             self.binr = cv2.threshold (self.img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) [1]
    #             self.invert = cv2.bitwise_not(self.binr)

    #             self.hit_miss = cv2.morphologyEx(self.invert, cv2.MORPH_HITMISS, self.kernel)
    #             self.hit_miss = cv2.resize(self.hit_miss, (500, 500))
    #             cv2.imwrite(file_path + "\hit_miss.jpg", self.hit_miss)
    #             self.image2 = Image.open(file_path + "\hit_miss.jpg")
                
    #         elif index == 9:
    #             # Toán tử Rectange
    #             self.binr = cv2.threshold (self.img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) [1]
    #             self.invert = cv2.bitwise_not(self.binr)

    #             self.rect = cv2.morphologyEx(self.invert, cv2.MORPH_RECT, self.kernel)
    #             self.rect = cv2.resize(self.rect, (500, 500))
    #             cv2.imwrite(file_path + "\_rect.jpg", self.rect)
    #             self.image2 = Image.open(file_path + "\_rect.jpg")
                
    #         elif index == 10:
    #             # Toán tử Cross
    #             self.binr = cv2.threshold (self.img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) [1]
    #             self.invert = cv2.bitwise_not(self.binr)

    #             self.cross = cv2.morphologyEx(self.invert, cv2.MORPH_CROSS, self.kernel)
    #             self.cross = cv2.resize(self.hit_miss, (500, 500))
    #             cv2.imwrite(file_path + "\cross.jpg", self.cross)
    #             self.image2 = Image.open(file_path + "\cross.jpg")
                
    #         self.photo2 = ImageTk.PhotoImage(self.image2.resize((500, 500), Image.Resampling.LANCZOS))
    #         self.canvas2.create_image(0, 0, anchor='nw', image=self.photo2)    
    #     except:
    #         messagebox.showerror("Có lỗi!", "Không thể mở hình ảnh.")

    # def select_image(self):
    #     self.image_path = filedialog.askopenfilename(filetypes=[('Image Files', '*.png; *.jpg; *.jpeg')])
    #     if not self.image_path:
    #         return
    #     try:
    #         # Mở ảnh được chọn và tạo một phiên bản thu nhỏ để hiển thị trên Canvas
    #         self.image = Image.open(self.image_path)
    #         self.photo = ImageTk.PhotoImage(self.image.resize((500, 500), Image.Resampling.LANCZOS))
    #         # Hiển thị ảnh trên Canvas
    #         self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
    #     except:
    #         messagebox.showerror("Có lỗi!", "Không thể mở hình ảnh.")
    #         self.image_path = None
    
    def clear_image(self):
        self.canvas.delete("all")
        self.canvas2.delete("all")
        self.image_path = None
        self.image = None
        self.photo = None
    
root = tk.Tk()
app = ImageSelector(master=root)
app.mainloop()
