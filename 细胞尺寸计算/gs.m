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

% 保存滤波后的图像
imwrite(filtered_img, 'p2.jpg');

P_signal = sum(p1(:).^2);
P_noise_denoised = sum((filtered_img(:) - p1(:)).^2);
SNR_denoised = 10*log10(P_signal/P_noise_denoised);
disp(SNR_denoised);