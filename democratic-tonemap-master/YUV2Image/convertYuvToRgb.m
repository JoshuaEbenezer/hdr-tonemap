function rgb = convertYuvToRgb(yuv)
% convert row vector YUV [0, 255] in row vector RGB [0, 255]

load conversion.mat; % load conversion matrices

yuv = double(yuv);

yuv(:, 2 : 3) = yuv(:, 2 : 3) - 512;
rgb = (yuvToRgb *yuv.').';

rgb = uint16(clipValue(rgb, 0, 1024));