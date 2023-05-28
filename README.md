# 信号与系统课程设计

[TOC]

## 图像处理(平滑、锐化与边缘提取)

### 理论基础

图像处理(image processing)，用计算机对图像进行分析，以达到所需结果的技术。又称影像处理。图像处理一般指数字图像处理。数字图像是指用工业相机、摄像机、扫描仪等设备经过拍摄得到的一个大的二维数组，该数组的元素称为像素，其值称为灰度值。图像处理技术一般包括图像压缩，增强和复原，匹配、描述和识别3个部分。

空间滤波是一种采用滤波处理的影像增强方法。目的是改善影像质量，包括去除高频噪声与干扰，及影像边缘增强、线性增强以及去模糊等。平滑和锐化滤波器是处理数字图像的常用方法。

平滑的主要目的是减少图像中的噪声。锐化的主要目的是突出图像中的细节或增强已经模糊的细节。

### 图像平滑处理

图像在获取、传输的过程中，可能会受到干扰的影响，会产生噪声，噪声是一种出错了的信号，噪声会造成图像粗糙，需要我们对图像进行平滑处理，保留有用的信号。

在imgpr1.m，imgpr2.m中我们使用均值滤波、高斯滤波、中值滤波对img1，img2进行了平滑处理，分别得到三种结果，由于img1中存在椒盐干扰，因此中值滤波效果最好，对于img2则三种效果相似。

img1（最左为原图）

![image-20230523232030867](https://zjyimage.oss-cn-beijing.aliyuncs.com/202305232320952.png)

img2（最左为原图）

![image-20230523231924542](https://zjyimage.oss-cn-beijing.aliyuncs.com/202305232319447.png)

### 图像锐化处理

图像锐化*(image sharpening)*是补偿图像的轮廓，增强图像的边缘及灰度跳变的部分，使图像变得清晰，图像锐化是为了突出图像上地物的边缘、轮廓，或某些线性目标要素的特征。这种滤波方法提高了地物边缘与周围像元之间的反差，因此也被称为边缘增强。

图像平滑往往使图像中的边界、轮廓变得模糊，为了减少这类不利效果的影响，这就需要利用图像锐化技术，使图像的边缘变的清晰。

公式推导：
$$
▽^2f=▽f(x+1)−▽f(x)\\
▽f(x)=f(x+1)−f(x)\\
$$
可得:    
$$
f(x)=f(x+1)−▽f(x)\\
$$
则：
$$
f(x)=f(x+1)−▽f(x)+▽^2f(x)\\
$$
那么锐化后的图像即为:
$$
g(x)=f(x+1)−▽f(x)+k▽^2f(x)\\
$$
在不考虑精确度的情况下:
$$
g(x)=f(x)+k▽^2f(x)
$$
在imgpr3.m中我们使用两种不同卷积核对img3进行了锐化处理（最左为原图）：

![image-20230523232048348](https://zjyimage.oss-cn-beijing.aliyuncs.com/202305232320462.png)

### 图像边缘提取

在imgpr4.m中，我们利用高斯滤波和canny算子，对img2进行了边缘提取：

![image-20230523232154263](https://zjyimage.oss-cn-beijing.aliyuncs.com/202305232321314.png)

## 求图像中细胞平均半径

### 原图

![image](https://zjyimage.oss-cn-beijing.aliyuncs.com/202305282026738.jpeg)

### 预处理

#### 中值滤波

图中很明显存在椒盐噪声（黑白点），因此首先使用中值滤波进行处理。

处理后：

![p1](https://zjyimage.oss-cn-beijing.aliyuncs.com/202305282026731.jpeg)

#### 高斯低通滤波与巴特沃斯低通滤波

之后使用傅里叶变换，将其转换到频域上，再进行后续操作：

- 高斯低通滤波

```matlab
% 读入原始图像
img = imread('p1.jpg');

% 对原始图像进行傅里叶变换
F = fftshift(fft2(img));

% 高斯低通滤波器参数
D0 = 50;

% 构造高斯低通滤波器
[x, y] = meshgrid(-(size(img,1)/2):(size(img,1)/2-1), -(size(img,2)/2):(size(img,2)/2-1));
dist = sqrt(x.^2 + y.^2);
gauss_filter = exp(-dist.^2./(2*D0^2));

% 滤波
filtered_F = F .* gauss_filter;

% 反变换
filtered_img = uint8(real(ifft2(ifftshift(filtered_F))));

% 显示滤波前后的图像
figure;
subplot(1,2,1);imshow(img);title('原始图像');
subplot(1,2,2);imshow(filtered_img);title('滤波后的图像');
```

得到图像![untitled](https://zjyimage.oss-cn-beijing.aliyuncs.com/202305282026807.png)

- 巴特沃斯低通滤波

```matlab
p1 = imread('p1.jpg');
F=double(p1);%数据类型转换，MATLAB不支特图像的无符号整型的i计算
G = fft2(F);%傅立叶变换
G=fftshift(G);%转换数据矩阵
[M,N]=size(G);
nn=2;%二阶巴特沃斯(Butterworth)高通滤波器
d0=30;
m=fix (M/2);n=fix(N/2);
for i=1:M
    for j=1:N
        d=sqrt((i-m)^2+(j-n)^2);
        h=1/(1+0.414*(d/d0)^(2*nn));%i计算传递函数
        result(i,j) = h*G(i,j);
    end
end
result=ifftshift(result);
Y2=ifft2(result);
Y3=uint8(real(Y2));
subplot(121),imshow(p1),title('原图像');%滤波后图像显示
subplot(122),imshow(Y3),title('巴特沃斯低通滤波后图像');%滤波后图像显示

P_signal = sum(p1(:).^2);
P_noise_denoised = sum((Y3(:) - p1(:)).^2);
SNR_denoised = 10*log10(P_signal/P_noise_denoised);
disp(SNR_denoised);
```

得到图像![untitled](https://zjyimage.oss-cn-beijing.aliyuncs.com/202305282026799.png)

- 两种滤波的比较

  | 滤波   | 高斯低通滤波 | 巴特沃斯低通滤波 |
  | ------ | ------------ | ---------------- |
  | 信噪比 | 6.3678       | 5.1209           |

  因此选择高斯低通滤波，之后进行逆傅里叶变换

#### 其余预处理

1. 灰度化
2. 直方图均衡化
3. 低帽变换
4. 调节灰度对比
5. 二值化
6. 形态学操作
   - 开操作
   - 闭操作

### 计算细胞半径

#### hough圆形检测

```matlab
% 使用Hough变换进行圆形检测
[centers, radii] = imfindcircles(BW3,[4, 20]);

% 计算细胞平均半径
avgRadius = mean(radii);
sigma = std(radii);
lower_threshold = avgRadius - 2 * sigma;
upper_threshold = avgRadius + 2 * sigma;

% 找到位于阈值内的所有元素
idx = find(radii >= lower_threshold & radii <= upper_threshold);
radii = radii(idx);
centers = centers(idx,:);
avgRadius = mean(radii);

figure;
% 显示原始图像和检测结果
imshow(p2); hold on;
h = viscircles(centers, radii,'EdgeColor','b');
title(['Average cell radius: ', num2str(avgRadius)]);
```

结果如图所示：

![untitled](https://zjyimage.oss-cn-beijing.aliyuncs.com/202305282026826.png)

#### bwlabel函数求半径

```matlab
% 连通区域分析，计算每个细胞的面积和周长，并计算等效直径
[L, num] = bwlabel(BW3);
stats = regionprops(L, {'Area', 'Perimeter'});
diameters = zeros(num, 1);
for i = 1 : num
    diameters(i) = 2 * sqrt(stats(i).Area / pi);
end

boundaries = bwboundaries(BW3);
disp(['The average diameter of cells is: ', num2str(avg_diameter)]);

figure
% 画出每个细胞的轮廓
imshow(BW3);
hold on;
for i=1:length(boundaries)
    boundary = boundaries{i};
    plot(boundary(:,2), boundary(:,1),'r','LineWidth',2);
end
```

结果如图所示

![untitled](https://zjyimage.oss-cn-beijing.aliyuncs.com/202305282026872.png)

#### 两种结果对比

1. hough圆形检测法，对于部分形状不是圆的细胞，可能并不适用；
2. bwlabel函数求半径法，由于很多细胞重叠在一起，导致被认定为一个细胞，因此结果比较差。
