function [gfull] = gen_GRD_waveform(k,grd_detail,gmax,smax,dt)
%[gfull] = gen_GRD_waveform(k,grd_detail,gmax,smax,dt)
%   Generates full GRD waveforms based on the excitation k-space and number
%   of samples passed on grd_detail. This code assumes blips are slew
%   constrained as they are tipically very short and does not check if
%   maximum gradient amplitude is exceeded. The gradient blips are also
%   designed to take their whole allocated duration rather than minimizing
%   their duration by enforcing maximum slew rate.
%   INPUTS:
%       k             -> excitation k-space matrix (dimensions: Naxis x 
%                     Nblips) [rad/m]
%       grd_detail    -> matrix detailing number of samples in 1st column
%                     and sample type in 2nd column (No GRD = 0, GRD = 1)
%       gmax          -> maximum gradient amplitude (NOT USED - can provide 
%                     dummy value) 
%       smax          -> maximum slew rate [mT/m/ms]
%       dt            -> dwell time [seconds]
%   OUTPUTS:
%       gfull         -> matrix with full GRD waveform (dimensions: 
%                     Nsamples x Naxis) [mT/m]
%
% David Leitao 2024, King's College London (david.leitao@kcl.ac.uk)

gamma_mT = 267522.119; % units rad s^-1 mT^-1
garea = diff([k,zeros(3,1)],[],2) /gamma_mT; %[mT/m*s]

Nsamples = sum(grd_detail(:,1));
gfull = zeros(Nsamples,3);

n = 1; blip = 0;
while n<size(grd_detail,1)
    if grd_detail(n,2)==1
        blip = blip + 1;
        idx = sum(grd_detail(1:n-1,1))+1:sum(grd_detail(1:n,1));
        %design normalized triangular blip and scale wrt required area
        gtrap = gen_trap_normalized(numel(idx));
        %check slew constraint
        gareamax = 1e3*dt*smax * sum(gtrap)/abs(min(diff(gtrap))) * dt; %[mT/m*s]
        if gareamax<max(abs(garea(:,blip)))
            error('Cannot achieve desired gradient blip #%d within slew constraints',blip)
        end
        gfull(idx,:) = gtrap*garea(:,blip)'/dt;
    end
    n = n + 1;
end


    function [trapezoid_norm] = gen_trap_normalized(Nsamples)
        Nsamples = Nsamples + 2; %to accomodate samples with zero
        if mod(Nsamples,2)==0 %even no. of samples
            trapezoid_norm = linspace(0,1,Nsamples/2).';
            trapezoid_norm = cat(1,trapezoid_norm,flip(trapezoid_norm,1));
        else %odd no. of samples
            trapezoid_norm = linspace(0,1,(Nsamples+1)/2).';
            trapezoid_norm = cat(1,trapezoid_norm,flip(trapezoid_norm(1:end-1),1));
        end
        trapezoid_norm([1 end]) = []; %truncate zero samples
        %normalise to unit area:
        trapezoid_norm = trapezoid_norm ./sum(trapezoid_norm); 
    end
end