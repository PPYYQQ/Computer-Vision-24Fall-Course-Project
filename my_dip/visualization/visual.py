import wandb

wandb.init(project='DIP', name='denoise_snail_log')

log_file_path = 'denoise_snail_log.txt'

with open(log_file_path, 'r') as log_file:
    for line in log_file:
        parts = line.strip().split()
        if len(parts) < 6:
            continue
        iteration = int(parts[1])
        loss = float(parts[3])
        psnr_noisy = float(parts[5])
        psrn_gt = float(parts[7])
        psnr_gt_sm = float(parts[9])
        wandb.log({
            'Iteration': iteration,
            'Loss': loss,
            'PSNR_noisy': psnr_noisy,
            'PSRN_gt': psrn_gt,
            'PSNR_gt_sm': psnr_gt_sm
        })
wandb.finish()