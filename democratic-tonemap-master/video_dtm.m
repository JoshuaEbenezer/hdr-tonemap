folder = '/media/labuser-admin/seagate/hdr_videos/yuv/';
files = dir(fullfile(folder,'*.yuv'));
for k=1:length(files)
    basefilename = files(k).name;
    filename = fullfile(folder,basefilename);
    [~, baseFileNameNoExt, ~] = fileparts(fullFileName);
end
    
% 
% outimbase = '/media/labuser-admin/seagate/hdr_videos/kmeans_tmo/acropolis_104_';
% outimext='.png';
% imids = 1:331;
% 
% v = VideoWriter('newfile.avi','Motion JPEG AVI');
% 
% % parameters
% nrims = length(imids);
% nrbins = 5000; % nr of bins in the histogram
% K = 256; % nr of output bins
% wc = 0.7; % weighting between the two different color models
% mf = 5000; % the multiplication factor in log transform
% nbrfwbw = 3; % the number of frames forward and backward used in the keyframe histogram estimation
% imstack_length = 2*nbrfwbw+1;
% interp_length = 20; % the numebr of frames between key frames 
% w = 3840;
% h = 2160;
% 
% 
% im=double(loadFileYuv(filename,w,h,1));
% %im=pfs_read_image([imbase sprintf('%3.3d',imids(1)) imext]);
% maxfact = double(mf/max(im(:)));
% 
% [N,M,C]=size(im);
% imseq = zeros(N,M,C,imstack_length,'single');
% for iii = 1:imstack_length,
%     imseq(:,:,:,iii)= double(loadFileYuv(filename,w,h,iii));
%     %imseq(:,:,:,iii)=pfs_read_image([imbase sprintf('%3.3d',imids(iii)) imext]);
% end
% 
% [hh,bb,maxy1] = histkeyframe(imseq,nrbins,maxfact);
% [c_1,id_1]=dtm_nopa_histxin(hh,bb,K,maxy1);
% county = 0;
% 
% for iii = 1:nbrfwbw,
%     im = imseq(:,:,:,iii);
%     maxy = double(log(1+maxfact*max(im(:))));
%     c_i = c_1;
%     id_i = dtm_nn_from_c(c_i,nrbins,maxy);
%     [im1,im2]=dtm_toneim2(double(im),id_i,maxfact);
%     outim = wc*im1+(1-wc)*im2;
%     imwrite(uint8(outim),[outimbase num2str(imids(iii)) outimext]);
%     disp(['Finished frame nr ' num2str(imids(iii))])
% end
% 
% 
% imid = 1+nbrfwbw;
% while imid<=(nrims-interp_length-nbrfwbw)
%     for iii = -nbrfwbw:nbrfwbw                                                    
%         imseq(:,:,:,nbrfwbw+1+iii)=double(loadFileYuv(filename,w,h,imids(iii+interp_length+imid)));
%         %imseq(:,:,:,nbrfwbw+1+iii)=pfs_read_image([imbase sprintf('%3.3d',imids(iii+interp_length+imid)) imext]);
%     end
%     [hh,bb,maxy2] = histkeyframe(imseq,nrbins,maxfact);
%     [c_2,id_2]=dtm_nopa_histxin(hh,bb,K,maxy2);
%     for iii = 1:interp_length
%         %im = pfs_read_image([imbase sprintf('%3.3d',imids(iii+imid-1)) imext]);
%         im = double(loadFileYuv(filename,w,h,imids(iii+imid-1)));
%         w1 = (interp_length+1-iii)/(interp_length);
%         w2 = (iii-1)/(interp_length);
%         maxy = double(log(1+maxfact*max(im(:))));
%         if w2==0,
%             c_i = c_1;
%         else
%             c_i = w1*c_1+w2*c_2;
%         end
%         id_i = dtm_nn_from_c(c_i,nrbins,maxy);
%         [im1,im2]=dtm_toneim2(double(im),id_i,maxfact);
%         outim = wc*im1+(1-wc)*im2;
%         imwrite(uint8(outim),[outimbase num2str(imids(iii+imid-1)) outimext]);
%   
%         disp(['Finished frame nr ' num2str(imids(iii+imid-1))])
%         county = county+1;
%     end
%     c_1 = c_2;
%     id_1 = id_2;
%     maxy1 = maxy2;
%     imid = imid + interp_length;
% end
% 
% nrleft = nrims-imid+1;
% 
% if imid<=nrims,
%     for iii = -nbrfwbw:nbrfwbw                                                    ,
%         imseq(:,:,:,nbrfwbw+1+iii)=hdrread([imbase sprintf('%3.3d',imids(nrims-nbrfwbw+iii)) imext]);
%         %imseq(:,:,:,nbrfwbw+1+iii)=pfs_read_image([imbase sprintf('%3.3d',imids(nrims-nbrfwbw+iii)) imext]);
%     end
%     [hh,bb,maxy2] = histkeyframe(imseq,nrbins,maxfact);
%     [c_2,id_2]=dtm_nopa_histxin(hh,bb,K,maxy2);
%     for iii = 1:nrleft,
%         %im = pfs_read_image([imbase sprintf('%3.3d',imids(iii+imid-1)) imext]);
%         im = hdrread([imbase sprintf('%3.3d',imids(iii+imid-1)) imext]);
%         w1 = (nrleft+1-iii)/(nrleft);
%         w2 = (iii-1)/(nrleft);
%         maxy = double(log(1+maxfact*max(im(:))));
%         if w2==0,
%             c_i = c_1;
%         else
%             c_i = w1*c_1+w2*c_2;
%         end
%         id_i = dtm_nn_from_c(c_i,nrbins,maxy);
%         [im1,im2]=dtm_toneim2(double(im),id_i,maxfact);
%         outim = wc*im1+(1-wc)*im2;
%         
%         imwrite(uint8(outim),[outimbase num2str(imids(iii+imid-1)) outimext]);
%         disp(['Finished frame nr ' num2str(imids(iii+imid-1))])
%     end
% end