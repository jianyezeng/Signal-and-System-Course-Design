p2 = imread('p2.jpg');
figure;
subplot(2,3,1);imshow(p2);title('滤波后图像');

p_gray = im2gray(p2);
subplot(2,3,2);imshow(p_gray);title('滤波后图像');

p_heq=adapthisteq(p_gray,'NumTiles',[10,10]);
subplot(2,3,3);imshow (p_heq);title('直方图均衡化后图像')

se=strel('disk',20);%构造S
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


% 连通区域分析，计算每个细胞的面积和周长，并计算等效直径
[L, num] = bwlabel(BW3);
stats = regionprops(L, {'Area', 'Perimeter'});
diameters = zeros(num, 1);
for i = 1 : num
    diameters(i) = 2 * sqrt(stats(i).Area / pi);
end
avg_diameter = mean(diameters);
boundaries = bwboundaries(BW3);
disp(['The average diameter of cells is: ', num2str(avg_diameter)]);

figure
% 画出每个细胞的轮廓
imshow(p2);
hold on;
for i=1:length(boundaries)
    boundary = boundaries{i};
    plot(boundary(:,2), boundary(:,1),'r','LineWidth',2);
end