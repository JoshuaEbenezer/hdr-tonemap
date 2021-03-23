% function mov = loadFileYuv(fileName, width, height, idxFrame)

function [imgRgb] = loadFileYuv(fileId, width, height, framenum)
% load RGB movie [0, 255] from YUV 4:2:0 file


subSampleMat = [1, 1; 1, 1];


% search fileId position
sizeFrame = 3 * width * height;
fseek(fileId, (framenum - 1) * sizeFrame, 'bof');

% read Y component
buf = fread(fileId, width * height, 'uint16');
imgYuv(:, :, 1) = reshape(buf, width, height).'; % reshape

% read U component
buf = fread(fileId, width / 2 * height / 2, 'uint16');
imgYuv(:, :, 2) = kron(reshape(buf, width / 2, height / 2).', subSampleMat); % reshape and upsample

% read V component
buf = fread(fileId, width / 2 * height / 2, 'uint16');
imgYuv(:, :, 3) = kron(reshape(buf, width / 2, height / 2).', subSampleMat); % reshape and upsample

% normalize YUV values
% imgYuv = imgYuv / 255;

% convert YUV to RGB
imgRgb = reshape(convertYuvToRgb(reshape(imgYuv, height * width, 3)), height, width, 3);
% imgRgb = ycbcr2rgb(imgYuv);
%imwrite(imgRgb,'ActualBackground.bmp','bmp');
% mov(f) = im2frame(imgRgb);
% 	mov(f).cdata = uint8(imgRgb);
% 	mov(f).colormap =  [];
%     imwrite(imgRgb,'ActualBackground.bmp','bmp');

%figure, imshow(imgRgb);
%name = 'ActualBackground.bmp';
%Image = imread(name, 'bmp');
%figure, imshow(Image);

