p2 = imread('p2.jpg');
figure;
subplot(2,3,1);imshow(p2);title('滤波后图像');

p_gray = im2gray(p2);
subplot(2,3,2);imshow(p_gray);title('灰度化后图像');

p_heq=adapthisteq(p_gray,'NumTiles',[10,10]);
subplot(2,3,3);imshow (p_heq);title('直方图均衡化后图像')

se=strel('disk',10);%构造S
p_th=imbothat(p_heq,se);%底帽变换，去除不均匀背景
subplot (2,3,4);imshow (p_th);title('底帽变换后');

p_agc = imadjust(p_th);
subplot(2,3,5);imshow(p_agc);title('调节灰度对比后');

level=graythresh(p_agc);%获取灰度图像gray的阈值
BW=imbinarize(p_agc,level);%将灰度图像gray转化为二值图像bw
subplot(2,3,6);imshow (BW);title('二值化后');

figure;
% 对二值化图像进行形态学操作
SE = strel('disk', 2);
BW2 = imopen(BW, SE);
BW3 = imclose(BW2, SE);
imshow(BW3);title('形态学操作后');

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