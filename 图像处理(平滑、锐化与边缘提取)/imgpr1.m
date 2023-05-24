p1 = imread('img1.jpg');
figure;
%均值滤波
filter1 = 1/9*ones(3);
p11 = imfilter(p1,filter1);

%带权均值滤波（高斯滤波）
filter2 = 1/16*[1,2,1;2,4,2;1,2,1];
p12 = imfilter(p1,filter2);

%中值滤波，可有效解决椒盐噪声
%图一主要存在椒盐噪声，因此该效果最好
p13 = medfilt2(p1,[3 3]);

subplot(141);imshow(p1);
subplot(142);imshow(p11);
subplot(143);imshow(p12);
subplot(144);imshow(p13);