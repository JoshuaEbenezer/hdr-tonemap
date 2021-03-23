function rgb = convertYuvToRgb(yuv)
% convert row vector YUV [0, 255] in row vector RGB [0, 255]

load conversionbt2020.mat; % load conversion matrices

yuv = double(yuv);

yuv(:, 2 : 3) = yuv(:, 2 : 3) - 512;
rgb = (yuv2rgb_bt2020 *yuv.').';

rgb = uint16(clipValue(rgb, 0, 1024));