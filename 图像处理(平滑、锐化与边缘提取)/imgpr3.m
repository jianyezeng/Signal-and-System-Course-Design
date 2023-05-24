p = imread('img3.jpg');
figure;
filter1 = [0,1,0;1,-4,1;0,1,0];
filter2 = [-1,-1,-1,-1,-1;-1,1,1,1,-1;-1,1,9,1,-1;-1,1,1,1,-1;-1,-1,-1,-1,-1];
p1 = imfilter(p,filter1);
p2 = imfilter(p,filter2);
subplot(231);imshow(p);
subplot(232);imshow(p1);
subplot(235);imshow(p+p1)
subplot(233);imshow(p2);
subplot(236);imshow(p2+p);