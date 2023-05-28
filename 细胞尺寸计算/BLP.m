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