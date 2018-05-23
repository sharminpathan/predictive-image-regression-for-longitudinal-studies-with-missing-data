% 
% histgram normalization
%
close all
clear all

addpath('../plot-code');
ims{1} = loadMETA('mri_1_a.mhd');
ims{2} = loadMETA('mri_2_a.mhd');
ims{3} = loadMETA('mri_3_a.mhd');

figure;
for i = 1:length(ims)
    subplot(3, 3, i), imagesc(ims{i}); colormap(gray); axis equal; axis off
end

for i = 1:length(ims)
    tmp = ims{i};
    minI = min(tmp(:));
    maxI = max(tmp(:));
    tmp = (tmp - minI)/(maxI - minI);
    idx = find(tmp~=0);
    tmp(idx) = histeq(tmp(idx), 256);
    ims{i} = tmp;
end

for i = 1:length(ims)
    subplot(3, 3, i+3), imagesc(ims{i}); colormap(gray); axis equal; axis off
    subplot(3, 3, i+6), hist(ims{i}, 100);
end
