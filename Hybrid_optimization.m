%% Hybrid optimization of kT-points
%%% This script optimizes a kT-points pulse using the hybrid cost function. 
%%% Both RF and gradients are optimized using a multi-start strategy as 
%%% described in the paper, under the STA regime.
%%%
%%% For computational performance, the following code makes use of:
%%%  -parallelization: 4 cores by default, adjustable
%%%  -clustering: voxels are clustered using k-means based on their B1, B0 and coordinates
%%%  -analytical first and second order derivatives
%%%
%%% Any bugs/queries contact: david.leitao@kcl.ac.uk
%%% David Leitao @ King's College London 2024

clearvars; close all; clc; 

addpath('./mat')
addpath('./lib')

%update number of workers
if isempty(gcp('nocreate'))
    c = parcluster('local');
    c.NumWorkers = min(4,maxNumCompThreads);
    parpool(c, c.NumWorkers);
end

%% Define sequence/pulse parameters for optimization

gamma_uT = 267.522119; %gyromagnetic ratio [rad /s /uT]

%p1 and p2 depend on the RF waveform shapes (see eqs.[8,9] in the paper)
p1 = 1; % = 1 for a rectangular pulse
p2 = 2; % = 1 for a rectangular pulse
beta_min =@(fa,tr,tau) p2*deg2rad(fa)/(gamma_uT*p1*sqrt(tr*tau)); %minimum B1rms (eq.[12] in the paper) 

%%% Sequence & pulse parameters
TR           = 8e-3;   %repetiton time [seconds]
alpha_target = 15;     %target flip angle [degrees]
Nsp          = 5;      %number of RF subpulses in kT-points pulse
tau_rf       = 200e-6; %duration of RF subpulses [seconds]    
tau_grd      = 100e-6; %duration of gradient blips [seconds]
lambda       = 0.5;    %balancing parameter
beta_target  = 1.2*beta_min(alpha_target,TR,Nsp*tau_rf); %target B1rms [uT]

%%% Enable/disable clustering 
use_clusters = true;


%% Load adjustment and constraints data

load('AdjData.mat')
[xgrid,ygrid,zgrid] = ndgrid(adjdata.pos_PE,adjdata.pos_RO,adjdata.pos_SS); 
Npe = numel(adjdata.pos_PE);
Nro = numel(adjdata.pos_RO); 
Nsl = numel(adjdata.pos_SS);
Nch = size(adjdata.B1map,4); 
tx = 1e-3*reshape(adjdata.B1map,[],Nch); %[mT/V] 
b0 = 2*pi*adjdata.B0map(:)/(1e3*gamma_uT); %[mT]
pos = [xgrid(:), ygrid(:), zgrid(:)]; %[m]
mask = adjdata.mask_pdes;
Nvoxels = sum(mask(:));

%load hardware & SAR constraints and limits; note that average power is
%calculated using a VOP matrix notation (with only one non-zero element)
load('ConstraintData.mat')
ConstrLimits = cat(1,limits.lSAR*TR,...
                     limits.gSAR*TR, ...
                     limits.Pmax*TR, ...
                     limits.Vmax*ones(Nch*Nsp,1));


%perform k-means clustering of all voxels based on their coordinates, 
%transmit sensitivities and off-resonance - uses the centroid of each
%cluster for the optimisation weighted by their size
if use_clusters
    %vector with all properties
    X = [pos(mask(:),1), pos(mask(:),2), pos(mask(:),3), real(tx(mask(:),:)), imag(tx(mask(:),:)), b0(mask(:))];
    Xmean = mean(X,1); Xstd = std(X,[],1);
    Xnormal = (X - repmat(Xmean,[size(X,1) 1])) ./ repmat(Xstd,[size(X,1) 1]);

    Nclusters = 500;
    rng('default')
    %perform k-means, get cluster indices and centroids:
    [indices,C] = kmeans_pp(Xnormal.',Nclusters);
    pos_cluster =  C(1:3,:).'   .* repmat(Xstd(1:3),[Nclusters 1])   + repmat(Xmean(1:3),[Nclusters 1]);
    tx_cluster  = (C(4:11,:).'  .* repmat(Xstd(4:11),[Nclusters 1])  + repmat(Xmean(4:11),[Nclusters 1])) + ...
               1i*(C(12:19,:).' .* repmat(Xstd(12:19),[Nclusters 1]) + repmat(Xmean(12:19),[Nclusters 1]));
    b0_cluster =  C(20,:).'    .* repmat(Xstd(20),[Nclusters 1])    + repmat(Xmean(20),[Nclusters 1]);
    weights_cluster = ones(Nclusters,1);
    for cluster_idx = 1:Nclusters
        cluster_voxels = find(indices==cluster_idx);
        weights_cluster(cluster_idx) = numel(cluster_voxels);
    end
    weights_cluster = weights_cluster / mean(weights_cluster);
    fprintf(1,'%d voxels compressed into %d clusters via k-means.\n',Nvoxels,Nclusters);
end


%% Setup optimization structure and run optimization

if use_clusters
    alpha_des = deg2rad(alpha_target) * ones(Nclusters,1); %[rad]
    beta_des  = sqrt(TR/(Nsp*tau_rf)) * beta_target * ones(Nclusters,1); %[uT]; B1rms over pulse duration
    optim_struct.pos = pos_cluster;
    optim_struct.b0 = b0_cluster;
    optim_struct.tx = tx_cluster;
    optim_struct.weights = weights_cluster;
else
    alpha_des = deg2rad(alpha_target) * ones(Nvoxels,1); %[rad] 
    beta_des  = sqrt(TR/(Nsp*tau_rf)) * beta_target * ones(Nvoxels,1); %[uT]; B1rms over pulse duration
    optim_struct.pos = pos(mask(:),:);
    optim_struct.b0 = b0(mask(:),:);
    optim_struct.tx = tx(mask(:),:);
    optim_struct.weights = ones(Nvoxels,1);
end

optim_struct.VOP               = VOP;
optim_struct.constraint_limits = ConstrLimits;
optim_struct.verbose           = 'off';
optim_struct.maxiter           = 1e2;
optim_struct.quality           = 2;
optim_struct.multistart        = true;
optim_struct.multistart_trials = 8;
optim_struct.smax              = 100; %[mT/m/ms]
optim_struct.gmax              = 30;  %[mT/m]
optim_struct.Nsp               = Nsp;
optim_struct.subpulse_length_s = tau_rf; %[s]
optim_struct.subpulse_gap_s    = tau_grd; %[s]
optim_struct.power_factor_s    = p2 * tau_rf; %for SAR calculations
optim_struct.x0                = [];

rng('default')
tic
[fval, b_opt, k_opt] = design_hybrid_kTpoints(alpha_des,beta_des,lambda,optim_struct);
fprintf(1,'kT-points optimization (RF and gradients) with %d sub-pulses finished in %.1f seconds.\n',Nsp,toc)

%% Compute and plot flip angle and B1rms maps

dt = 10e-6;
RFsamples = tau_rf/dt;
GRDsamples = tau_grd/dt;
rf_form   = repmat([ones(RFsamples,1); zeros(GRDsamples,1)],[Nsp 1]);  
rf_detail = repmat([RFsamples 1; GRDsamples 0],[Nsp 1]);  
RFfull  = gen_RF_waveform(b_opt,rf_form,rf_detail);
grd_detail = repmat([RFsamples 0; GRDsamples 1],[Nsp 1]);  
GRDfull  = gen_GRD_waveform(k_opt,grd_detail,optim_struct.gmax,optim_struct.smax,dt);

%%% Plot full optimized RF and GRD waveforms
figure('color','w','position',[200 250 800 375]); 
subplot(2,1,1)
plot((1:size(RFfull,1))*1e3*dt,abs(RFfull),'linewidth',1.5)
xlabel('t (ms)'); ylabel('RF magnitude (V)'); title('Optimized RF pulse')
legend([repmat('TX ch.',[Nch 1]),num2str((1:Nch)','%.0f')],'NumColumns',2); grid on; set(gca,'fontsize',12)

subplot(2,1,2)
plot((1:size(GRDfull,1))*1e3*dt,GRDfull,'linewidth',1.5)
xlabel('t (ms)'); ylabel('Gradient (mT/m)'); title('Optimized GRD waveform')
legend('G_{PE}','G_{RO}','G_{SS}'); grid on; set(gca,'fontsize',12)

%%% Compute flip angle
[mxy,mz] = blochsim_CK(RFfull,GRDfull,pos,tx,b0,'dt',dt);
mxymap = reshape(mxy,Npe,Nro,Nsl);
alpha = double(adjdata.mask_view) .* rad2deg(asin(abs(mxymap)));

figure('color','w','position',[200 450 1000 275]); 
subplot(1,3,1)
imagesc(1e3*adjdata.pos_PE,1e3*adjdata.pos_SS,squeeze(abs(alpha(:,Nro/2,:))),[0.5 1.5]*rad2deg(alpha_des(1))); axis image; hcb = colorbar; colormap('parula')
xlabel('SS direction (mm)'); ylabel('PE direction (mm)'); title('Transverse slice'); title(hcb,'deg')

subplot(1,3,2)
imagesc(1e3*flip(adjdata.pos_RO),1e3*adjdata.pos_SS,flip(squeeze(abs(alpha(Npe/2,:,:)))),[0.5 1.5]*rad2deg(alpha_des(1))); axis image; hcb = colorbar; colormap('parula')
xlabel('SS direction (mm)'); ylabel('RO direction (mm)'); title('Coronal slice'); title(hcb,'deg')

text(0.5,1.25,'Flip angle map','units','normalized','fontsize',14,'horizontalalignment','center','fontweight','bold')

subplot(1,3,3)
imagesc(1e3*adjdata.pos_RO,1e3*adjdata.pos_PE,rot90(squeeze(abs(alpha(:,:,Nsl/2)))),[0.5 1.5]*rad2deg(alpha_des(1))); axis image; hcb = colorbar; colormap('parula')
xlabel('PE direction (mm)'); ylabel('RO direction (mm)'); title('Sagittal slice'); title(hcb,'deg')


%%% Compute B1rms (over TR)
b1rms = sqrt(sum(abs(1e3*tx*RFfull.').^2,2)*dt/TR); %[uT]
b1rms = double(adjdata.mask_view) .* reshape(b1rms,Npe,Nro,Nsl);

figure('color','w','position',[200 75 1000 275]); 
subplot(1,3,1)
imagesc(1e3*adjdata.pos_PE,1e3*adjdata.pos_SS,squeeze(b1rms(:,Nro/2,:)),[0.5 1.5]*beta_des(1)*sqrt((Nsp*tau_rf)/TR)); axis image; hcb = colorbar; colormap('hot')
xlabel('SS direction (mm)'); ylabel('PE direction (mm)'); title('Transverse slice'); title(hcb,'\mu{}T')

subplot(1,3,2)
imagesc(1e3*flip(adjdata.pos_RO),1e3*adjdata.pos_SS,flip(squeeze(b1rms(Npe/2,:,:))),[0.5 1.5]*beta_des(1)*sqrt((Nsp*tau_rf)/TR)); axis image; hcb = colorbar; colormap('hot')
xlabel('SS direction (mm)'); ylabel('RO direction (mm)'); title('Coronal slice'); title(hcb,'\mu{}T')

text(0.5,1.25,'B1rms map','units','normalized','fontsize',14,'horizontalalignment','center','fontweight','bold')

subplot(1,3,3)
imagesc(1e3*adjdata.pos_RO,1e3*adjdata.pos_PE,rot90(squeeze(b1rms(:,:,Nsl/2))),[0.5 1.5]*beta_des(1)*sqrt((Nsp*tau_rf)/TR)); axis image; hcb = colorbar; colormap('hot')
xlabel('PE direction (mm)'); ylabel('RO direction (mm)'); title('Sagittal slice'); title(hcb,'\mu{}T')


