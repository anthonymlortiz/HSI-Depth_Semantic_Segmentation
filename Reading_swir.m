%@uthor: Anthony Ortiz
%Creation Date: 09/11/2017
%Last Modification date: 09/11/2017

%Reading SWIR Image using Matlab

clear 
close all

load('cropped_r_swir_reg.mat');
[rows, columns, depth]=size(cropped_r_swir_reg );
ground_truth = zeros(rows, columns, 3);

%Helpful for labeling
band_53 = cropped_r_swir_reg(:,:,53 );
figure,imagesc(band_53) ; axis image; axis off ; colormap hsv;
% The previous command will open a figure, using the data cursor is
% possible to inspect the X (column) and Y (row) of the pixels

%If the class is not clear from this image go to:
%https://www.google.com/maps/place/La+Brea+Tar+Pits+%26+Museum/@34.0639088,-118.3587605,678m/data=!3m2!1e3!4b1!4m5!3m4!1s0x80c2b922fdf520ff:0x74ce772b0af26299!8m2!3d34.0638079!4d-118.3554338
%on google maps, ther you will find an aerial RGB image for reference




%Assign color of the class to pixels in "ground_truth" while labeling
%For reference visit: https://www.mathworks.com/help/matlab/ref/colorspec.html
%Assigning color examples:

%Red
%ground_truth[i, j,1] = 255

%Green
%ground_truth[i, j,2] = 255

%Blue
%ground_truth[i, j,3] = 255

%White
%ground_truth[i, j,:] = 255

%Black
%DO NOTHING

%Yellow
%ground_truth[i, j,1] = 255
%ground_truth[i, j,2] = 255


