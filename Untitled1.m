i = imread('test.jpg');
i1 = imresize(i,[1536 2048]);
imwrite(i1,'testimage.jpg');