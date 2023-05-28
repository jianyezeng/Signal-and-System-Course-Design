p = imread('image.jpg');
p=im2gray(p);
figure;
%中值滤波
p1 = medfilt2(p,[3 3]);
subplot(111);imshow(p1);
imwrite(p1, 'p1.jpg','JPEG')