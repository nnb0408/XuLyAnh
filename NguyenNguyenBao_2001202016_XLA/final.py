import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk,font
from PIL import ImageTk, Image
import cv2, pathlib, os
import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import threshold_otsu
from skimage.feature import hog
from skimage.transform import resize




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
        self.cout = 1
        # slt_ic = PhotoImage(file="C:\\bao\\Python\\NguyenNguyenBao_2001202016_XLA\\Images_Data\\select.png")
        self.label0 = tk.Label(self, text="Nguyễn Nguyên Bảo - 2001202016", font=("Arial", 28, "bold"), fg = 'red')#, background=BLUE_COLOR)
        # self.label0.configure(background=BLUE_COLOR)
        self.label0.pack(side="top", padx=15, pady=10)

        self.button_frame = tk.Frame(self)
        self.button_frame.pack(side="top", fill="x")
        self.select_button = tk.Button(self.button_frame, text="Chọn Ảnh", fg="white", command=self.select_image) #image=slt_ic, compound='left')#, command=self.select_image)
        self.select_button.pack(side="left", padx=15, pady=15)
        self.clear_button = tk.Button(self.button_frame, text="Xóa Ảnh", fg="white", command=self.clear_image)#, command=self.clear_image)
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
        self.combo1 = ttk.Combobox(self.button_frame, state="readonly", values=['K-Mean', 'Phát Hiện Đường Viền', 'Ngưỡng Otsu', 'Dùng Mặt Nạ Màu'])
        self.combo1.set('Phân Đoạn Ảnh')
        self.combo1.pack(side="left", padx=25, pady=15)

        # Combobox thứ hai
        self.combo2 = ttk.Combobox(self.button_frame, state="readonly", values=['Dặc trưng HOG', 'Đặc trưng SIFT'])
        self.combo2.set('Đặc trưng ảnh')
        self.combo2.pack(side="left", padx=25, pady=15)

        # Combobox thứ ba
        self.combo3 = ttk.Combobox(self.button_frame, state="readonly", values=['Điều chỉnh độ sáng và độ tương phản','Làm sắc nét hình ảnh', 'Loại bỏ nhiễu khỏi hình ảnh', 'Tăng cường màu sắc trong hình ảnh','Biến đổi nghịch đảo','Cân bằng biểu đồ xám'])
        self.combo3.set('Nâng Cao Chất Lượng Ảnh')
        self.combo3.pack(side="left", padx=25, pady=15)

        # Combobox thứ tư
        self.combo4 = ttk.Combobox(self.button_frame, state="readonly", values=['Phép Co','Phép Giãn','Phép Mở','Phép Đóng','Morphological Gradient','Top Hat','Black Hat','Hit Miss', 'Rectange', 'Cross'])
        self.combo4.set('Xử Lý Hình Thái')
        self.combo4.pack(side="left", padx=25, pady=15)

        # # Combobox thứ năm
        # self.combo5 = ttk.Combobox(self.button_frame, state="readonly", values=['A','B','C'])
        # self.combo5.set('Phân ngưỡng ảnh')
        # self.combo5.pack(side="left", padx=30, pady=15)
        
        # Comboboxes
        self.combo1["font"] = ("Arial", 15) 
        self.combo2["font"] = ("Arial", 15) 
        self.combo3["font"] = ("Arial", 15) 
        self.combo4["font"] = ("Arial", 15)

        # self.combo1.bind('<<ComboboxSelected>>', self.on_select_combo1)
        # self.combo2.bind('<<ComboboxSelected>>', self.on_select_combo1)
        # self.combo3.bind('<<ComboboxSelected>>', self.on_select_combo1)
        # self.combo4.bind('<<ComboboxSelected>>', self.on_select_combo1)

        # Canvas thứ nhất chứa ảnh gốc
        self.canvas_frame = tk.Frame(self)
        self.canvas_frame.pack(side="left", fill="both", expand=True, padx=15)
        
        self.canvas = tk.Canvas(self.canvas_frame, width=480, height=450, background="light gray")
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

        # đăng ký sự kiện change của các combobox
        self.combo1.bind("<<ComboboxSelected>>", self.update_image_combobox1)
        self.combo2.bind("<<ComboboxSelected>>", self.update_image_combobox2)
        self.combo3.bind("<<ComboboxSelected>>", self.update_image_combobox3)
        self.combo4.bind("<<ComboboxSelected>>", self.update_image_combobox4)
        # self.combo5.bind("<<ComboboxSelected>>", self.update_image_combobox5)
        


    # Hàm xử lý sự kiện của combobox1
    def update_image_combobox1(self, event=None):
        if self.image_path is None:
            return

        try:
            # Tạo thư mục lưu kết quả
            p = pathlib.Path('./KQ_PDA')
            p.mkdir(exist_ok=True)
            file_path = os.path.abspath('./KQ_PDA')
            print(file_path)
            # Đọc ảnh vào
            img = cv2.imread(self.image_path)
            # Lấy index
            index = self.combo1.current() + 1
            # K_mean
            if index == 1:
                twoDimage = img.reshape((-1,3))
                twoDimage = np.float32(twoDimage)

                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                K = 4
                attempts = 10
                # Sử Dụng K-Mean            
                ret, label, center=cv2.kmeans (twoDimage, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
                center = np.uint8(center)
                res = center[label.flatten()]
                self.imagecbb1_1 = res.reshape((img.shape))

                cv2.imwrite(file_path + "\self.imagecbb1_1.jpg", self.imagecbb1_1)                
                self.image2 = Image.open(file_path + "\self.imagecbb1_1.jpg")

            # Phân đoạn hình ảnh bằng phát hiện đường viền
            if index == 2:
                img2 = cv2.resize(img,(500,300))

                gray = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                _,thresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)
                # Lấy ảnh đường viền
                self.imagecbb1_2 = cv2.dilate(cv2.Canny(thresh, 0, 255), None)

                cnt = sorted(cv2.findContours(self.imagecbb1_2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
                mask = np.zeros((300,500), np. uint8)
                self.imagecbb1_2 = cv2.drawContours (mask, [cnt], -1, 255, -1)

                dst = cv2.bitwise_and(img2, img2, mask=mask)
                self.imagecbb1_2 = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

                cv2.imwrite(file_path + "\self.imagecbb1_2.jpg", self.imagecbb1_2)                
                self.image2 = Image.open(file_path + "\self.imagecbb1_2.jpg")

            # Dùng ngưỡng Otsu
            if index == 3:
                img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
                thresh = threshold_otsu(img_gray)
                img_otsu = img_gray<thresh
                plt.imshow(img_otsu)

                def filter_image (image, mask):
                    r = image[:,:,0] * mask
                    g = image[:,:,1] * mask
                    b = image[:,:,2] * mask
                    return np.dstack([r,g,b])
                self.imagecbb1_3 = filter_image (img, img_otsu)

                cv2.imwrite(file_path + "\self.imagecbb1_3.jpg", self.imagecbb1_3)                
                self.image2 = Image.open(file_path + "\self.imagecbb1_3.jpg")

            # Dùng mặt nạ màu:
            if index ==4:
                low = np.array([0, 0, 0])
                high = np.array([200, 170, 170])
                mask = cv2.inRange(img, low, high)
                self.imagecbb1_4 = cv2.bitwise_and(img, img, mask=mask)

                cv2.imwrite(file_path + "\self.imagecbb1_4.jpg", self.imagecbb1_4)                
                self.image2 = Image.open(file_path + "\self.imagecbb1_4.jpg") 

            self.original_width, self.original_height = self.image2.size       
            
            scale_width = self.canvas2.winfo_width() / self.original_width       
            scale_height = self.canvas2.winfo_height() / self.original_height
                
            scale = min(scale_width, scale_height)
            
            self.photo2 = ImageTk.PhotoImage(self.image2.resize((int(self.original_width * scale),  
                int(self.original_height * scale)), Image.Resampling.LANCZOS))
                
            self.canvas2.create_image(0, 0, anchor='nw', image=self.photo2)
            # self.label0.config(text="Success {}".format(self.cout))
            self.cout +=1 
            

        except:
            messagebox.showerror("Có lỗi!", "Không thể mở hình ảnh.")
    
    # Hàm xử lý sự kiện của combobox2    
    def update_image_combobox2(self, event=None):
        if self.image_path is None:
            return
        try:
            # Tạo thư mục lưu kết quả
            p = pathlib.Path('./KQ_DC')
            p.mkdir(exist_ok=True)
            file_path = os.path.abspath('./KQ_DC')
                # Đọc ảnh vào
            img = cv2.imread(self.image_path)
            # Lấy index
            index = self.combo2.current() + 1

            # Dùng đặt trung HOG
            if index == 1:
                # Resize ảnh
                resized_img = resize(img, (64*4, 64*4))
                # resized_img = np.squeeze(resized_img)
                # print(resized_img.shape)
                _,self.imagecbb2_1 = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, channel_axis=2)

                cv2.imwrite(file_path + "\self.imagecbb2_1.jpg", self.imagecbb2_1)                
                self.image2 = Image.open(file_path + "\self.imagecbb2_1.jpg")

            # Dùng đăt trung SIFT
            if index == 2:
                #convert to greyscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                #create SIFT feature extractor
                sift = cv2.SIFT_create()
                #detect features from the image
                keypoints, descriptors = sift.detectAndCompute (img, None)
                #draw the detected key points
                self.imagecbb2_2 = cv2.drawKeypoints(gray, keypoints, img)

                cv2.imwrite(file_path + "\self.imagecbb2_2.jpg", self.imagecbb2_2)                
                self.image2 = Image.open(file_path + "\self.imagecbb2_2.jpg")

            self.original_width, self.original_height = self.image2.size       
            
            scale_width = self.canvas2.winfo_width() / self.original_width       
            scale_height = self.canvas2.winfo_height() / self.original_height
                
            scale = min(scale_width, scale_height)
            
            self.photo2 = ImageTk.PhotoImage(self.image2.resize((int(self.original_width * scale),  
                int(self.original_height * scale)), Image.Resampling.LANCZOS))
                
            self.canvas2.create_image(0, 0, anchor='nw', image=self.photo2)
            # self.label0.config(text="Success {}".format(self.cout))
            self.cout +=1            
        except:
            messagebox.showerror("Có lỗi!", "Không thể mở hình ảnh.")
    
    # Hàm xử lý sự kiện của combobox3    
    def update_image_combobox3(self, event=None):
        if self.image_path is None:
            return
        try:
            #Tạo thư mục lưu kết quả
            p = pathlib.Path('./KQ_NCCL')
            p.mkdir(exist_ok=True)
            file_path = os.path.abspath('./KQ_NCCL')
            #Lấy ảnh
            img = cv2.imread(self.image_path)
            index = self.combo3.current() + 1

            if index == 1:
                brightness = 10
                contrast = 2.3
                self.imagecbb3_1 = cv2.addWeighted(img, contrast, np.zeros(img.shape, img.dtype), 0, brightness)
                cv2.imwrite(file_path + "\imagecbb3_1.jpg", self.imagecbb3_1)                
                self.image2 = Image.open(file_path + "\imagecbb3_1.jpg")

            if index == 2:
                kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
                # Làm sắc nét hình ảnh
                self.imagecbb3_2 = cv2.filter2D(img, -1, kernel)
                # Làm sắc nét hình ảnh bằng toán tử Laplacian
                # self.imagecbb3_2 = cv2.Laplacian(img, cv2.CV_64F)

                cv2.imwrite(file_path + "\imagecbb3_2.jpg", self.imagecbb3_2)                
                self.image2 = Image.open(file_path + "\imagecbb3_2.jpg")

            if index == 3:
                # Loại bỏ nhiễu bằng bộ lọc Median   
                # self.imagecbb3_3 = cv2.medianBlur(img, 3)
                # Loại bỏ nhiễu bằng bộ lọc Gaussian
                self.imagecbb3_3 = cv2.GaussianBlur(img, (7,7), 0)

                cv2.imwrite(file_path + "\imagecbb3_3.jpg", self.imagecbb3_3)                
                self.image2 = Image.open(file_path + "\imagecbb3_3.jpg")
                
            if index == 4:
                # Chuyển đổi hình ảnh từ không gian màu BGR sang HSV
                image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                # Điều chỉnh màu sắc, độ bão hòa và giá trị của hình ảnh(1.5 - 1.5 - 1)
                # Điều chỉnh sắc độ bằng cách nhân nó với 1.5 
                image[:, :, 0] = image[:, :, 0] * 1.5
                # Điều chỉnh độ bão hòa bằng cách nhân nó với 1.5
                image[:, :, 1] = image[:, :, 1] * 1.5
                # Điều chỉnh giá trị bằng cách nhân nó với 1
                image[:,:, 2] = image[:, :, 2] * 1
                # Chuyển đổi hình ảnh trở lại không gian màu BGR
                self.imagecbb3_4 = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

                cv2.imwrite(file_path + "\imagecbb3_4.jpg", self.imagecbb3_4)                
                self.image2 = Image.open(file_path + "\imagecbb3_4.jpg")

            if index == 5:
                temp = Image.open(self.image_path)
                pixel = temp.load()
                for i in range(temp.size[0]):
                    for j in range(temp.size[1]):
                        x,y,z = pixel [i,j][0],pixel [i,j][1],pixel [i,j][2]
                        x,y,z = abs(x-255), abs(y-255), abs(z-255)
                        pixel [i,j] = (x,y,z)
                # Chuyển ảnh về numpy
                temp1 = np.array(temp)
                # Chuyển ảnh về dạng UMat        
                self.imagecbb3_5 = cv2.UMat(temp1)

                cv2.imwrite(file_path + "\imagecbb3_5.jpg", self.imagecbb3_5 )                
                self.image2 = Image.open(file_path + "\imagecbb3_5.jpg")

            if index == 6:
                #Chuyển thành ảnh xám
                gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Cân bằng biểu đồ xám
                self.imagecbb3_6 = cv2.equalizeHist(gray_image)

                cv2.imwrite(file_path + "\imagecbb3_6.jpg", self.imagecbb3_6 )                
                self.image2 = Image.open(file_path + "\imagecbb3_6.jpg")

            self.original_width, self.original_height = self.image2.size       
            
            scale_width = self.canvas2.winfo_width() / self.original_width       
            scale_height = self.canvas2.winfo_height() / self.original_height
                
            scale = min(scale_width, scale_height)
            
            self.photo2 = ImageTk.PhotoImage(self.image2.resize((int(self.original_width * scale),  
                int(self.original_height * scale)), Image.Resampling.LANCZOS))
                
            self.canvas2.create_image(0, 0, anchor='nw', image=self.photo2)
            # self.label0.config(text="Success {}".format(self.cout))
            self.cout +=1 
        except:
            messagebox.showerror("Có lỗi!", "Không thể mở hình ảnh.")

    # Hàm xử lý sự kiện của combobox4    
    def update_image_combobox4(self, event=None):
        if self.image_path is None:
            return
        # try:
        # Tạo thư mục lưu kết quả
        p = pathlib.Path('./KQ_XLHT')
        p.mkdir(exist_ok=True)
        file_path = os.path.abspath('./KQ_XLHT')

        # Đọc ảnh vào
        img = cv2.imread(self.image_path,0)
        # Tạo ma trận có kích thước 10x10
        self.kernel = np.ones((10,10), np.uint8)
        # Áp dụng các phép với kernel đã tạo 

        self.combo2.set('Phân Ngưỡng')
        self.combo3.set('Nâng Cao Chất Lượng')
        self.combo1.set('Phân Đoạn Ảnh')

        index = self.combo4.current() + 1

        if index == 1:
            # Phép co:
            self.erosion = cv2.erode(img, self.kernel, iterations = 1)
            self.erosion = cv2.resize(self.erosion, (500, 500))
            cv2.imwrite(file_path + "\erosion.jpg", self.erosion)                
            self.image2 = Image.open(file_path + "\erosion.jpg")

        elif index == 2:
            # Phép giãn:
            self.dilation = cv2.dilate(img, self.kernel, iterations = 1)
            self.dilation = cv2.resize(self.dilation, (500, 500))
            cv2.imwrite(file_path + "\dilation.jpg", self.dilation)
            self.image2 = Image.open(file_path + "\dilation.jpg")
        
        elif index == 3:
            # Phép mở:
            self.opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, self.kernel)
            self.opening = cv2.resize(self.opening, (500, 500))
            cv2.imwrite(file_path + "\opening.jpg", self.opening)
            self.image2 = Image.open(file_path + "\opening.jpg")

        
        elif index == 4:
            # Phép đóng:
            self.closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, self.kernel)
            self.closing = cv2.resize(self.closing, (500, 500))
            cv2.imwrite(file_path + "\closing.jpg", self.closing)
            self.image2 = Image.open(file_path + "\closing.jpg")

        elif index == 5:
            # Morphological Gradient
            self.gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, self.kernel)
            self.gradient = cv2.resize(self.gradient, (500, 500))
            cv2.imwrite(file_path + "\gradient.jpg", self.gradient)
            self.image2 = Image.open(file_path + "\gradient.jpg")

        elif index == 6:
            # Toán tử Top Hat
            self.tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, self.kernel)
            self.tophat = cv2.resize(self.tophat, (500, 500))
            cv2.imwrite(file_path + "\_tophat.jpg", self.tophat)
            self.image2 = Image.open(file_path + "\_tophat.jpg")

        elif index == 7:
            # Toán tử Black Hat
            self.blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, self.kernel)
            self.blackhat = cv2.resize(self.blackhat, (500, 500))
            cv2.imwrite(file_path + "\_blackhat.jpg", self.blackhat)
            self.image2 = Image.open(file_path + "\_blackhat.jpg")
            
        elif index == 8:
            # Toán tử Hit Miss
            self.binr = cv2.threshold (img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) [1]
            self.invert = cv2.bitwise_not(self.binr)

            self.hit_miss = cv2.morphologyEx(self.invert, cv2.MORPH_HITMISS, self.kernel)
            self.hit_miss = cv2.resize(self.hit_miss, (500, 500))
            cv2.imwrite(file_path + "\hit_miss.jpg", self.hit_miss)
            self.image2 = Image.open(file_path + "\hit_miss.jpg")
            
        elif index == 9:
            # Toán tử Rectange
            self.binr = cv2.threshold (img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) [1]
            self.invert = cv2.bitwise_not(self.binr)

            self.rect = cv2.morphologyEx(self.invert, cv2.MORPH_RECT, self.kernel)
            self.rect = cv2.resize(self.rect, (500, 500))
            cv2.imwrite(file_path + "\_rect.jpg", self.rect)
            self.image2 = Image.open(file_path + "\_rect.jpg")
            
        elif index == 10:
            # Toán tử Cross
            self.binr = cv2.threshold (img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) [1]
            self.invert = cv2.bitwise_not(self.binr)

            self.cross = cv2.morphologyEx(self.invert, cv2.MORPH_CROSS, self.kernel)
            self.cross = cv2.resize(self.hit_miss, (500, 500))
            cv2.imwrite(file_path + "\cross.jpg", self.cross)
            self.image2 = Image.open(file_path + "\cross.jpg")
            
        self.original_width, self.original_height = self.image2.size       
            
        scale_width = self.canvas2.winfo_width() / self.original_width       
        scale_height = self.canvas2.winfo_height() / self.original_height
            
        scale = min(scale_width, scale_height)
        
        self.photo2 = ImageTk.PhotoImage(self.image2.resize((int(self.original_width * scale),  
            int(self.original_height * scale)), Image.Resampling.LANCZOS))
            
        self.canvas2.create_image(0, 0, anchor='nw', image=self.photo2)
        # self.label0.config(text="Success {}".format(self.cout))
        self.cout +=1 
    
        # except:
        #     messagebox.showerror("Có lỗi!", "Không thể mở hình ảnh.")

    def select_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[('Image Files', '*.png; *.jpg; *.jpeg')])
        if not self.image_path:
            return
        try:
            self.image = Image.open(self.image_path)
            self.original_width, self.original_height = self.image.size       
            
            scale_width = self.canvas.winfo_width() / self.original_width       
            scale_height = self.canvas.winfo_height() / self.original_height
                
            scale = min(scale_width, scale_height)
            
            self.photo = ImageTk.PhotoImage(self.image.resize((int(self.original_width * scale),  
                int(self.original_height * scale)), Image.Resampling.LANCZOS))
                
            self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        except:
            messagebox.showerror("Có lỗi!", "Không thể mở hình ảnh.")
            self.image_path = None
    
    def clear_image(self):
        self.canvas.delete("all")
        self.canvas2.delete("all")
        self.image_path = None
        self.image = None
        self.photo = None
    
root = tk.Tk()
app = ImageSelector(master=root, background=BLUE_COLOR)
app.create_widgets()
app.mainloop()
