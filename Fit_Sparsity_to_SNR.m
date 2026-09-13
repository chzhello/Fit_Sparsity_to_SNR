%% Normalize for sparsity
clear all
clc
file  = fullfile(pwd, 'train_sparisty.xlsx');
num_edge_train=xlsread(file,1);
file1 = fullfile(pwd, 'train_sparisty_tru_snr.xlsx');
tru_snr=xlsread(file1,1);

tru_snr(tru_snr == 65535 | isnan(tru_snr)) = NaN;
nan_rows_union = any(isnan(tru_snr), 2);
tru_snr(nan_rows_union, :) = NaN;

tru_snr_mean = nanmean(tru_snr);
phi = nanmean(num_edge_train);
minphi=min(phi);
maxphi=max(phi);
uniform_phi=(phi-min(phi))./(max(phi)-min(phi));


%% Plotting the fitted curves for Sparsity and SNR
num_edge_mean_train=uniform_phi;
fitModel = fittype('poly7');
% Generate a model of the fitted function
fittedModel = fit(num_edge_mean_train',tru_snr_mean', fitModel);
% Generate data points for plotting
x_values = linspace(min(num_edge_mean_train), max(num_edge_mean_train), 100);
y_fit = feval(fittedModel, x_values);


%Plotting the fitted curves for Sparsity and SNR
figure(1);
scatter(num_edge_mean_train, tru_snr_mean, 'b','b', 'Marker', 'o', 'LineWidth', 2);
hold on;
plot(x_values, y_fit, 'r-','LineWidth',2,'MarkerSize',10);
hold off
set(gca,'FontSize',16)
grid
leggd = legend('Raw Dataset', 'Polynomial Fit');
leggd.FontSize = 16;
ylabel('$\bar{\eta}_{\mbox{lo}}$','interpreter','latex','FontSize',20);
xlabel('$\bar{\Psi}_\ell$','interpreter','latex','FontSize',20);
grid on;

%Goodness of fit of the poly7 fit for the SparSNR mapping
y_obs = tru_snr_mean';                
y_hat = feval(fittedModel, num_edge_mean_train');  
resid = y_obs - y_hat;
valid = ~isnan(resid);
N_fit = sum(valid);
RMSE_train = sqrt(mean(resid(valid).^2));
SS_res = sum(resid(valid).^2);
SS_tot = sum((y_obs(valid) - mean(y_obs(valid))).^2);
R2_train = 1 - SS_res/SS_tot;
fprintf('Training:  RMSE = %.3f dB,  R^2 = %.4f  (N = %d)\n', RMSE_train, R2_train, N_fit);


%% Auxiliary functions...
% ME VAD definition function
function [vad_labels,val_precent] = meVAD(mel_energy)
    % Parameter setting
    Nnoise = 10; % Noise frames
    gamma = 0.001; % Control parameters
    S_m = [];
    N_m = [];
    
    % Calculate the first threshold
    mel_energy = sum(mel_energy, 1);
    En = mean(mel_energy(:, 1:Nnoise), 2);
    Emax = max(mel_energy, [], 2);
    eta_apr = min(1.2 * En, (Emax + En) / 2);
    vad_labels = mel_energy > eta_apr;

    mel_energy_n = mel_energy(Nnoise+1: length(mel_energy));
    vad_labels_n = vad_labels(Nnoise+1: length(vad_labels));
    S_m = mel_energy_n(vad_labels_n == 1);
    N_m = mel_energy_n(vad_labels_n == 0);
   
    % Estimated provisional snr and a posteriori thresholds    
    S_hat = sum(S_m) - sum(N_m);
    SNR_temp = 10 * log10(S_hat / sum(N_m));
    Emax_n = max(mel_energy(1:Nnoise));
    N_hat = Emax_n / (1 + gamma * SNR_temp);
    
    % Calculate a posteriori threshold
    eta_aps = min(1.2 * N_hat, (Emax + N_hat) / 2);
    
    % Apply a posteriori thresholding for speech/noise classification
    vad_labels = mel_energy > eta_aps;
   
    % Calculated probability values
    val_precent = 1 ./ (1 + exp(- 10000000 * (mel_energy - eta_aps)));
   
end


% Functions to construct noise-added signals
function [x_noisy, noise]=add_noise_model(d,SNR_dB)
    %d is a clean voice signal
    sigPower = sum(abs(d).^2)/length(d);
    noisePower=sigPower/(10^(SNR_dB/10));
    randn('seed',5);
    n=randn(length(d),1);
    n=n-mean(n);
    n=n/std(n);  
    noise=sqrt(noisePower).*n;
    x_noisy=noise+d;
end


% Calculate the sparsity function of an audio
function [num_edge_train]=Calculating_sparsity(signal_test, vad_labels)
    quantiz_level =8;
    overlap_length = 200;
    transmit_prob_mat_G = zeros(quantiz_level);
    for i_frame_start = 1:length(vad_labels)              
         if vad_labels(i_frame_start)==1
            x = signal_test((i_frame_start-1)*overlap_length+1:overlap_length*i_frame_start); % Subsections by window length
            x_quantiz = quantiz((x-min(x))./(max(x)-min(x)),0:1/quantiz_level:1); % Normalize[0,1]
            x_quantiz(x_quantiz==0)=1;   
            for l = 1:overlap_length-1
                transmit_prob_mat_G(x_quantiz(l),x_quantiz(l+1)) = transmit_prob_mat_G(x_quantiz(l),x_quantiz(l+1)) +1 ;
            end
            transmit_prob_mat_G(transmit_prob_mat_G~=0)=1; % Composition of the adjacency matrix
            num_edge(i_frame_start) = sum(sum(transmit_prob_mat_G))./(quantiz_level*quantiz_level);%???
            transmit_prob_mat_G = zeros(quantiz_level);
         else
            num_edge(i_frame_start) = 0;
         end
    end 
    num_edge_train = mean(num_edge(vad_labels==1));  
end

