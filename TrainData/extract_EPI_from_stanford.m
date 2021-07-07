clear; clc; close all; warning('off');
%% Settings: "scene(X).disp" makes sure your EPIs obbey the same occlusion relationship, if your method is depth-related. "scene(1).border" ensure we don't extract EPIs with too much black background.
scene = struct('name','Amethyst'); scene(1).border = [81,100,81,100]; scene(1).disp = [0, 1];
scene(2).name='Bracelet'; scene(2).border = [70,47,140,47]; scene(2).disp = [1, 1];
scene(3).name='Eucalyptus Flowers'; scene(3).border = [115,55,190,55]; scene(3).disp = [0, 1];
scene(4).name='JellyBeans'; scene(4).border = [130,50,112,100]; scene(4).disp = [0, 1];
scene(5).name='Knights'; scene(5).border = [0,0,0,0]; scene(5).disp = [1, 1];
scene(6).name='LegoBulldozer'; scene(6).border = [66,100,250,56]; scene(6).disp = [0, 1];
scene(7).name='rabbit'; scene(7).border = [124,127,72,80]; scene(7).disp = [0, 1];
scene(8).name='Tarot Cards'; scene(8).border = [0,0,0,0]; scene(8).disp = [1, 1];
scene(9).name='TarotcardsLD'; scene(9).border = [0,0,0,0]; scene(9).disp = [1, 1];
scene(10).name='Treasure Chest'; scene(10).border = [90,132,180,40]; scene(10).disp = [0, 1];
scene(11).name='Truck'; scene(11).border = [32,142,80,80]; scene(11).disp = [0, 1];


folder = '..\Datasets\LFdata\Stanford\';  % Your Stanford LF Archive's path
savepath = '.\StanfordEPI\';
downscale = 1;
stride_spa = 16;
ang_LF_original = 17;
ang_LF = 17;
%%
g_count=0;
for s = 1:length(scene)
    filepaths = dir(fullfile(folder,scene(s).name,'*.png'));
    count = 0;
    % Extract EPIs from one angular dimension
    for load_start = 1 : ang_LF_original : length(filepaths)-ang_LF_original
        load_end = load_start + ang_LF-1;
        k=0;
        for i=load_start:load_end
            k=k+1;
            im=imread(fullfile(folder,scene(s).name,filepaths(i).name));
            im = im(scene(s).border(1)+1:end-scene(s).border(2),scene(s).border(3)+1:end-scene(s).border(4),:);
            im = imresize(im, downscale);
            LF(:,:,:,k)=im;
        end
        [hei,wid,chl,angRes]=size(LF);
        k=0;
        for h = 1:stride_spa:hei
            count = count + 1;
            g_count = g_count + 1;
            k=k+1;
            EPI = permute(LF(h,:,:,:),[4,2,3,1]);
            if scene(s).disp(1) == 0
                EPI = flip(EPI,2);
            end
%             figure(1);imshow(EPI);
            imwrite(EPI, fullfile(savepath,[scene(s).name,'_EPI_',sprintf('%03d',count),'.png']));
        end
        clear LF;
    end
    
    % Extract EPIs from the other angular dimension
    for load_start = 1 : ang_LF_original
        load_end = ang_LF_original * (ang_LF-1) + load_start;
        k=0;
        for i=load_start:ang_LF_original:load_end
            k=k+1;
            im=imread(fullfile(folder,scene(s).name,filepaths(i).name));
            im = im(scene(s).border(1)+1:end-scene(s).border(2),scene(s).border(3)+1:end-scene(s).border(4),:);
            im = imresize(im, downscale);
            LF(:,:,:,k)=im;
        end
        [hei,wid,chl,angRes]=size(LF);
        k=0;
        for h = 1:stride_spa:wid
            count = count + 1;
            g_count = g_count + 1;
            k=k+1;
            EPI = permute(LF(:,h,:,:),[4,1,3,2]);
%             figure(1);imshow(EPI);
            imwrite(EPI, fullfile(savepath,[scene(s).name,'_03EPI_',sprintf('%03d',count),'.png']));
        end
        clear LF;
    end
     fprintf('%d/%d processed. Extracted %d examples\n',s, length(scene), g_count);
end
