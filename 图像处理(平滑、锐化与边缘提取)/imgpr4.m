p = imread('img2.jpg');
p=rgb2gray(p);

%带权均值滤波（高斯滤波）
filter2 = 1/16*[1,2,1;2,4,2;1,2,1];

ph = imfilter(p,filter2);
img_edge2 = edge(p,'canny');
figure;
imshow(img_edge2);title('高斯');
