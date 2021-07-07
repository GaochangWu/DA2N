clear;clc;close all;warning('off')
%% Settings: "scene(1).border" ensure we don't extract EPIs with too much black background.
scene = struct('name','torch'); scene(1).border = [182,4,8,8];
scene(2).name='thinplant'; scene(2).border = [182,7,12,12];
scene(3).name='decoration'; scene(3).border = [0,0,0,0];
scene(4).name='plant'; scene(4).border = [0,0,0,0];
scene(5).name='scarecrow'; scene(5).border = [0,0,0,0];
scene(6).name='trunks'; scene(6).border = [0,0,0,0];
scene(7).name='orchid'; scene(7).border = [0,0,0,0];
scene(8).name='africa'; scene(8).border = [0,0,0,0];
scene(9).name='ship'; scene(9).border = [0,0,0,0];
scene(10).name='dragon'; scene(10).border = [0,0,0,0];
scene(11).name='basket'; scene(11).border = [9,9,18,18];

folder = '..\Datasets\igl_ULF\';  % Your Unstructured LFs' path
savepath = '.\FineTuneData\';
stride_hei = 31;
stride_ang = 400;
ang_Res4EPI = 600;    % Setting angular resolution of extracted EPIs
%%
g_count=0;
for s = 1 : length(scene)
    filepaths = dir(fullfile(folder,scene(s).name,'*.jpg'));
    count = 0;
    for load_start = 1 : stride_ang : length(filepaths)-ang_Res4EPI
        load_end = load_start + ang_Res4EPI-1;
        k=0;
        for i=load_start:load_end
            k=k+1;
            im=imread(fullfile(folder,scene(s).name,filepaths(i).name));
            LF(:,:,:,k)=im(scene(s).border(1)+1:end-scene(s).border(2),scene(s).border(3)+1:end-scene(s).border(4),:);
        end
        [hei,wid,chl,angRes]=size(LF);
        k=0;
        for h = 1:stride_hei:hei
            count = count + 1;
            g_count = g_count + 1;
            k=k+1;
            EPI = permute(LF(h,:,:,:),[4,2,3,1]);
            figure(1);imshow(EPI);
%             imwrite(EPI, fullfile(savepath,[scene(s).name,'_EPI_',sprintf('%03d',count),'.png']));
        end
        clear LF;
    end
end
