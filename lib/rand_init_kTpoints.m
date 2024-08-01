function [x0] = rand_init_kTpoints(base_voltage,Nsp,Nch,smax,gmax,subpulses_gap)
%[x0] = rand_init_kTpoints(base_voltage,Nsp,Nch,smax,gmax,subpulses_gap)
%   Generates random solution of complex voltages and excitation k-space
%   trajectory using the same convention as design_hybrid_kTpoints.m
%   INPUTS:
%       base_voltage  -> maximum voltage amplitude used to generate random
%                     solutions
%       Nsp           -> no. of RF/gradient blips in kT-points pulse
%       Nch           -> no. of transmit channels
%       smax          -> maximum slew rate of the system [mT/m/ms]
%       gmax          -> maximum gradient amplitude of the system [mT/m]
%       subpulses_gap -> gradient blip duration [s]
%   OUTPUTS:
%       x0            -> column array containing real/imaginary parts of RF
%                     voltages and excitation k-space trajectory
%
% David Leitao 2024, King's College London (david.leitao@kcl.ac.uk)

gamma_uT = 267.522119; % units rad/s/uT

x0_rf_mag = base_voltage * rand(Nsp*Nch,1);
x0_rf_pha = exp(1i*2*pi*rand(Nsp*Nch,1));
x0_rf = x0_rf_mag(:) .* x0_rf_pha(:);

%slew limited maximum gradient amplitude for the gradient blip duration:
gmax_slew = smax*subpulses_gap/2; %[mT/m]
if gmax_slew>gmax
    Gmax = gmax; %maximum gradient limited by gradient amplitude
else
    Gmax = gmax_slew; %maximum gradient limited by slew
end
%integral of gradient blip at maximum amplitude
garea_max = Gmax * subpulses_gap/2; %[mT.s/m]
%compare against 7T wavelength: 8/m (V. Gras et al DOI 10.1016/j.jmr.2015.10.017)
if (1e3*gamma_uT)*garea_max/2/pi > 8
    garea_max = 8*2*pi/gamma_uT;
end

garea = (2*rand(Nsp,3)-1) * garea_max;
x0_k = 1e3*gamma_uT * flip(cumsum(flip(garea,1),1),1);
x0_k = x0_k(:);

x0 = [real(x0_rf); imag(x0_rf); x0_k];

end