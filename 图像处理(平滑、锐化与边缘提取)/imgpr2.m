p = imread('img2.jpg');
p=rgb2gray(p);
subplot(221);imhist(p);title('灰度直方图');
figure;
%均值滤波
filter1 = 1/9*ones(3);
p11 = imfilter(p,filter1);

%带权均值滤波（高斯滤波）
filter2 = 1/16*[1,2,1;2,4,2;1,2,1];
p12 = imfilter(p,filter2);

%中值滤波
p13 = medfilt2(p,[3 3]);
subplot(141);imshow(p);
subplot(142);imshow(p11);
subplot(143);imshow(p12);
subplot(144);imshow(p13);
