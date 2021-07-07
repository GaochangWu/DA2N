clear;close all;clc;
%% settings
folder = './FineTuneEPIs/';  % ./StanfordEPIs/  or  ./FineTuneEPIs/
savepath = './trainFT';    % ./train    or    ./trainFT
ang_in = 11; % ang_out = (ang_in-1)*scale+1
wid_in = 72;
stride_wid = 92;  % 20 for train, 92 for fine-tune
stride_ang = 23;  % 1 for train, 23 for fine-tune
scale = 3;
%% initialization
data = zeros((ang_in-1)*scale+1, wid_in, 1, 1);
label = zeros(1, 1, 1, 1);
%% generate data
filepaths = dir(fullfile(folder,'*.png'));

count = 0;
for i = 1 : length(filepaths)
    EPI = imread(fullfile(folder,filepaths(i).name));
    EPI = rgb2ycbcr(EPI); 
    EPI = im2double(EPI(:, :, 1));

    [~, wid, C] = size(EPI);
    if wid>=wid_in
        for y = 1 : stride_wid : wid-wid_in+1
            im_label = EPI(:, y : y + wid_in-1, :);
            [cur_ang, cur_wid, C] = size(im_label);
            
            for x = 1 : stride_ang : cur_ang-((ang_in-1)*scale+1)+1
                count = count+1;
                    data(:, :, 1:C, count) = im_label(x : x+(ang_in-1)*scale+1-1, :, :);
                    label(:, :, :, count) = 0;
%                     figure(1);imshow(im_label(x : x+(size_inputV-1)*scale+1-1, :, :));
            end
        end
        fprintf('%d/%d processed. Extracted %d examples\n',i, length(filepaths), count);
    end
end

order = randperm(count);
data = data(:, :, :, order);
label = label(:, :, :, order);
%% writing to HDF5
chunksz = 28;

created_flag = false;
totalct = 0;
for batchno = 1:floor(count/chunksz)
    last_read=(batchno-1)*chunksz;
    batchdata = data(:,:,:,last_read+1:last_read+chunksz);
    batchlabs = zeros(1,1,1,chunksz);
    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5([savepath,'.h5'], batchdata, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp([savepath,'.h5']);
