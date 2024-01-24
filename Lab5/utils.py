import math
from operator import pos
import imageio
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw
from scipy import signal
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from PIL import Image, ImageDraw

def add_border(x, color, pad=1):
    w = x.size()[1]
    nc = x.size()[0]
    px = Variable(torch.zeros(3, w+2*pad+30, w+2*pad))
    if color == 'red':
        px[0] =0.7 
    elif color == 'green':
        px[1] = 0.7
    if nc == 1:
        for c in range(3):
            px[c, pad:w+pad, pad:w+pad] = x
    else:
        px[:, pad:w+pad, pad:w+pad] = x

    # print(px.shape)
    return px

def draw_text_tensor(tensor, text):
    np_x = tensor.transpose(0, 1).transpose(1, 2).data.cpu().numpy()
    pil = Image.fromarray(np.uint8(np_x*255))
    draw = ImageDraw.Draw(pil)
    draw.text((4, 64), text, (0,0,0))
    img = np.asarray(pil)
    return Variable(torch.Tensor(img / 255.)).transpose(1, 2).transpose(0, 1)

def show_result(epoch, loss, mse, kld, tfr, beta, psnr, args):
    
    
    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx() 
    ax1.plot(list(range(epoch)), loss, 'r-', label='loss')
    ax1.plot(list(range(epoch)), mse, 'g-', label='mse')
    ax1.plot(list(range(epoch)), kld, 'b-', label='kld')
    ax2.plot(list(range(epoch)), tfr, 'c--', label='tfr')
    ax2.plot(list(range(epoch)), beta, 'y--', label='beta')

    plt.title('loss and ratio curve')
    ax1.set_xlabel('Epoch')    
    ax1.set_ylabel('Loss')  
    ax2.set_ylabel('Ratio')  
    ax1.legend()
    ax2.legend()
    fig.tight_layout()
    plt.savefig('./' + args.log_dir + '/loss_ratio.png')

    plt.close(fig)
    plt.title('psnr')
    
    plt.plot(list(range(0, epoch + 1, 5)), psnr, label='psnr')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.legend()
    plt.savefig('./' + args.log_dir + '/psnr.png')

def plot_pred_test(x, cond, modules, args, device):
    modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
    modules['posterior'].hidden = modules['posterior'].init_hidden()

    # x = x.permute(1, 0, 2, 3 ,4).to(device)
    # cond = cond.permute(1, 0, 2).to(device)
    # x.shape = (sequence_size, batch_size, channel, weight or height, height or weight)
    # cond.shape = (sequence_size, batch_size, data) = (sequence_size, batch_size, 7)

    

    

    nsample = 6
    generate_sequence = [[] for i in range(nsample)]
    # ground truth
    for i in range(args.n_past + args.n_future):
        generate_sequence[0].append(x[i])

    for s in range(1, nsample - 1):

        x_0 = x[0]
        pred_sequence = []
        pred_sequence.append(x_0)
        hidden = [modules['encoder'](x[i]) for i in range(args.n_past+args.n_future)]  

        for i in range(1, args.n_past + args.n_future):

            h_t = hidden[i][0].detach()    # each elememt of hidden includes (hidden code, [4 feature maps])
            
            if args.last_frame_skip or i < args.n_past:	
                h_t_minus_1, skip = hidden[i-1]
            else:
                h_t_minus_1 = hidden[i-1][0]
            h_t_minus_1 = h_t_minus_1.detach()

            if s == 1:
                # posterior
                if i < args.n_past:
                    z_t, _, _ = modules['posterior'](hidden[i][0])      # hidden[i][0] = hidden code
                    modules['frame_predictor'](torch.cat([cond[i-1], h_t_minus_1, z_t], 1)) 
                    pred_sequence.append(x[i])
                else: 
                    z_t, _, _ = modules['posterior'](hidden[i][0])      # hidden[i][0] = hidden code
                    g_t = modules['frame_predictor'](torch.cat([cond[i-1], h_t_minus_1, z_t], 1)).detach()
                    x_pred = modules['decoder']([g_t, skip]).detach()
                    hidden[i] = modules['encoder'](x_pred)
                    pred_sequence.append(x_pred)
            else: 

                if i < args.n_past:
                    z_t, _, _ = modules['posterior'](hidden[i][0])      # hidden[i][0] = hidden code
                    modules['frame_predictor'](torch.cat([cond[i-1], h_t_minus_1, z_t], 1)) 
                    pred_sequence.append(x[i])
                else: 
                    z_t = torch.randn(args.batch_size, args.z_dim).to(device)
                    g_t = modules['frame_predictor'](torch.cat([cond[i-1], h_t_minus_1, z_t], 1)).detach()
                    x_pred = modules['decoder']([g_t, skip]).detach()
                    hidden[i] = modules['encoder'](x_pred)
                    pred_sequence.append(x_pred)
            
            # x_pred.shape = (batch_size, channel, width or height, height or width)
        generate_sequence[s] = pred_sequence
    
    # find best
    best_psnr = 0
    best_id = 0
    for s in range(2, 5):
        _, _, psnr = finn_eval_seq(x[args.n_past:], generate_sequence[s][args.n_past:])
        ave_psnr = np.mean(np.concatenate(psnr))
        # print('ave: ', ave_psnr)
        # print('best: ', best_psnr)
        if ave_psnr > best_psnr:
            best_psnr = ave_psnr
            best_id = s
    generate_sequence[5] = generate_sequence[best_id]

    # png for best
    # for each data inthe batch
    for i in range(x_pred.shape[0]):
        # for each data sequence
        image_sequence_png = [generate_sequence[5][j][i].cpu() for j in range(args.n_past+args.n_future)]
        # image_sequence_gif = [generate_sequence[5][j][i].permute(1, 2, 0).cpu() for j in range(args.n_past+args.n_future)]
        # print(image_sequence_png[i].shape)
        image = make_grid(image_sequence_png, nrow=12)
        # print(image.shape)
        img_path = './' + args.model_path + '/' + str(i) + '.png' 
        save_image(image, img_path)
        # gif_path = './' + args.model_path + '/' + str(i) + '.gif' 
        # imageio.mimsave(gif_path, image_sequence_gif, duration=0.1)

    # gif
    # for each data inthe batch
    for i in range(x_pred.shape[0]):
        # for each data sequence
        gifs = [[] for j in range(args.n_past+args.n_future)]
        text = [[] for j in range(args.n_past+args.n_future)]
        gif_grid = []
        for s  in range(nsample):
            for j in range(args.n_past+args.n_future):

                    if s == 0:
                        # ground truth
                        img = add_border(x[j][i], 'green')
                        img = draw_text_tensor(img, 'Ground\ntruth')
                        
                        # gifs[j].append(add_border(x[j][i], 'green'))
                        # text[j].append('Ground\ntruth')
                    elif s ==1:
                        if j < args.n_past:
                            # posterior
                            img = add_border(generate_sequence[1][j][i], 'green')
                        else: 
                            img = add_border(generate_sequence[1][j][i], 'red')
                        img = draw_text_tensor(img, 'Approx.\nposterior')
                    
                    elif s == 2 or s == 3 or s == 4:

                        if j < args.n_past:
                            # random sample
                            img = add_border(generate_sequence[s][j][i], 'green')
                        else: 
                            img = add_border(generate_sequence[s][j][i], 'red')
                        img = draw_text_tensor(img, 'Random\nsample %d' %(s - 1))

                        
                    else:

                        if j < args.n_past:
                            # best SSIM
                            img = add_border(generate_sequence[5][j][i], 'green')
                        else: 
                            img = add_border(generate_sequence[5][j][i], 'red')
                        img = draw_text_tensor(img, 'Best SSIM')
                        

                    gifs[j].append(img)

        for j in range(args.n_past+args.n_future):
            gif_grid.append(make_grid(gifs[j], nrow=6).permute(1, 2, 0).cpu())
        
        gif_path = './' + args.model_path + '/' + str(i) + '.gif' 
        imageio.mimsave(gif_path, gif_grid, duration=0.2)
        

        # image_sequence_png = [generate_sequence[5][j][i].cpu() for j in range(args.n_past+args.n_future)]
        # image_sequence_gif = [generate_sequence[5][j][i].permute(1, 2, 0).cpu() for j in range(args.n_past+args.n_future)]
        # # print(image_sequence_png[i].shape)
        # image = make_grid(image_sequence_png, nrow=12)
        # # print(image.shape)
        # img_path = './' + args.model_path + '/' + str(i) + '.png' 
        # save_image(image, img_path)
        # gif_path = './' + args.model_path + '/' + str(i) + '.gif' 
        # imageio.mimsave(gif_path, image_sequence_gif, duration=0.1)

    

def pred_test(x, cond, modules, args, device):
    # initialize the hidden state.
    modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
    modules['posterior'].hidden = modules['posterior'].init_hidden()

    # x.shape = (batch_size, sequence_size, channel, weight or height, height or weight)
    # cond.shape = (batch_size, sequence_size, data) = (batch_size, sequence_size, 7)
    # permute: Returns a view of the original tensor input with its dimensions changed.
    # x = x.permute(1, 0, 2, 3 ,4).to(device)
    # cond = cond.permute(1, 0, 2).to(device)
    # x.shape = (sequence_size, batch_size, channel, weight or height, height or weight)
    # cond.shape = (sequence_size, batch_size, data) = (sequence_size, batch_size, 7)
    x_0 = x[0]
    pred_sequence = []
    pred_sequence.append(x_0)
    hidden = [modules['encoder'](x[i]) for i in range(args.n_past+args.n_future)]  
    for i in range(1, args.n_past + args.n_future):
        # print(i)

        # Tensor.detach()
        # Returns a new Tensor, detached from the current graph.
        # The result will never require gradient.
        h_t = hidden[i][0].detach()    # each elememt of hidden includes (hidden code, [4 feature maps])
        
        if args.last_frame_skip or i < args.n_past:	
            h_t_minus_1, skip = hidden[i-1]
        else:
            h_t_minus_1 = hidden[i-1][0]
        h_t_minus_1 = h_t_minus_1.detach()

        if i < args.n_past:
            z_t, _, _ = modules['posterior'](hidden[i][0])      # hidden[i][0] = hidden code
            modules['frame_predictor'](torch.cat([cond[i-1], h_t_minus_1, z_t], 1)) 
            pred_sequence.append(x[i])
        else: 
            z_t = torch.randn(args.batch_size, args.z_dim).to(device)
            g_t = modules['frame_predictor'](torch.cat([cond[i-1], h_t_minus_1, z_t], 1)).detach()
            x_pred = modules['decoder']([g_t, skip]).detach()
            hidden[i] = modules['encoder'](x_pred)
            pred_sequence.append(x_pred)

            # print(f'pre_de')
    return pred_sequence
def plot_pred(x, cond, modules, epoch, args, device):
    modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
    modules['posterior'].hidden = modules['posterior'].init_hidden()

    # x = x.permute(1, 0, 2, 3 ,4).to(device)
    # cond = cond.permute(1, 0, 2).to(device)
    # x.shape = (sequence_size, batch_size, channel, weight or height, height or weight)
    # cond.shape = (sequence_size, batch_size, data) = (sequence_size, batch_size, 7)
    x_0 = x[0]
    pred_sequence = []
    pred_sequence.append(x_0)
    hidden = [modules['encoder'](x[i]) for i in range(args.n_past+args.n_future)]  
    for i in range(1, args.n_past + args.n_future):

        h_t = hidden[i][0].detach()    # each elememt of hidden includes (hidden code, [4 feature maps])
        
        if args.last_frame_skip or i < args.n_past:	
            h_t_minus_1, skip = hidden[i-1]
        else:
            h_t_minus_1 = hidden[i-1][0]
        h_t_minus_1 = h_t_minus_1.detach()

        if i < args.n_past:
            z_t, _, _ = modules['posterior'](hidden[i][0])      # hidden[i][0] = hidden code
            modules['frame_predictor'](torch.cat([cond[i-1], h_t_minus_1, z_t], 1)) 
            pred_sequence.append(x[i])
        else: 
            z_t = torch.randn(args.batch_size, args.z_dim).to(device)
            g_t = modules['frame_predictor'](torch.cat([cond[i-1], h_t_minus_1, z_t], 1)).detach()
            x_pred = modules['decoder']([g_t, skip]).detach()
            hidden[i] = modules['encoder'](x_pred)
            pred_sequence.append(x_pred)
            
            # x_pred.shape = (batch_size, channel, width or height, height or width)

    for i in range(x_pred.shape[0]):
        # for each data sequence
        image_sequence_png = [pred_sequence[j][i].cpu() for j in range(args.n_past+args.n_future)]
        image_sequence_gif = [pred_sequence[j][i].permute(1, 2, 0).cpu() for j in range(args.n_past+args.n_future)]
        # print(image_sequence_png[i].shape)
        image = make_grid(image_sequence_png, nrow=12)
        # print(image.shape)
        img_path = './' + args.log_dir + '/gen/epoch' + str(epoch) + '_' + str(i) + '.png' 
        save_image(image, img_path)
        gif_path = './' + args.log_dir + '/gen/epoch' + str(epoch) + '_' + str(i) + '.gif' 
        imageio.mimsave(gif_path, image_sequence_gif, duration=0.1)

def pred(x, cond, modules, epoch, args, device):
    # initialize the hidden state.
    modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
    modules['posterior'].hidden = modules['posterior'].init_hidden()

    # x.shape = (batch_size, sequence_size, channel, weight or height, height or weight)
    # cond.shape = (batch_size, sequence_size, data) = (batch_size, sequence_size, 7)
    # permute: Returns a view of the original tensor input with its dimensions changed.
    # x = x.permute(1, 0, 2, 3 ,4).to(device)
    # cond = cond.permute(1, 0, 2).to(device)
    # x.shape = (sequence_size, batch_size, channel, weight or height, height or weight)
    # cond.shape = (sequence_size, batch_size, data) = (sequence_size, batch_size, 7)
    x_0 = x[0]
    pred_sequence = []
    pred_sequence.append(x_0)
    hidden = [modules['encoder'](x[i]) for i in range(args.n_past+args.n_future)]  
    for i in range(1, args.n_past + args.n_future):
        # print(i)

        # Tensor.detach()
        # Returns a new Tensor, detached from the current graph.
        # The result will never require gradient.
        h_t = hidden[i][0].detach()    # each elememt of hidden includes (hidden code, [4 feature maps])
        
        if args.last_frame_skip or i < args.n_past:	
            h_t_minus_1, skip = hidden[i-1]
        else:
            h_t_minus_1 = hidden[i-1][0]
        h_t_minus_1 = h_t_minus_1.detach()

        if i < args.n_past:
            z_t, _, _ = modules['posterior'](hidden[i][0])      # hidden[i][0] = hidden code
            modules['frame_predictor'](torch.cat([cond[i-1], h_t_minus_1, z_t], 1)) 
            pred_sequence.append(x[i])
        else: 
            z_t = torch.randn(args.batch_size, args.z_dim).to(device)
            g_t = modules['frame_predictor'](torch.cat([cond[i-1], h_t_minus_1, z_t], 1)).detach()
            x_pred = modules['decoder']([g_t, skip]).detach()
            hidden[i] = modules['encoder'](x_pred)
            pred_sequence.append(x_pred)

            # print(f'pre_de')
    return pred_sequence

def kl_criterion(mu, logvar, args):
  # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= args.batch_size  
  return KLD
    
def eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            origin = gt[t][i]
            predict = pred[t][i]
            for c in range(origin.shape[0]):
                ssim[i, t] += ssim_metric(origin[c], predict[c]) 
                psnr[i, t] += psnr_metric(origin[c], predict[c])
            ssim[i, t] /= origin.shape[0]
            psnr[i, t] /= origin.shape[0]
            mse[i, t] = mse_metric(origin, predict)

    return mse, ssim, psnr

def mse_metric(x1, x2):
    err = np.sum((x1 - x2) ** 2)
    err /= float(x1.shape[0] * x1.shape[1] * x1.shape[2])
    return err

# ssim function used in Babaeizadeh et al. (2017), Fin et al. (2016), etc.
def finn_eval_seq(gt, pred):
    
    T = len(gt)
    bs = gt[0].shape[0]
    # print(pred[0][1].shape)
    # print(len(pred))
    # print(bs)
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            origin = gt[t][i].detach().cpu().numpy()
            predict = pred[t][i].detach().cpu().numpy()
            for c in range(origin.shape[0]):
                res = finn_ssim(origin[c], predict[c]).mean()
                if math.isnan(res):
                    ssim[i, t] += -1
                else:
                    ssim[i, t] += res
                psnr[i, t] += finn_psnr(origin[c], predict[c])
            ssim[i, t] /= origin.shape[0]
            psnr[i, t] /= origin.shape[0]
            mse[i, t] = mse_metric(origin, predict)

    return mse, ssim, psnr

def finn_psnr(x, y, data_range=1.):
    mse = ((x - y)**2).mean()
    return 20 * math.log10(data_range) - 10 * math.log10(mse)

def fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()

def finn_ssim(img1, img2, data_range=1., cs_map=False):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)

    K1 = 0.01
    K2 = 0.03

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    mu1 = signal.fftconvolve(img1, window, mode='valid')
    mu2 = signal.fftconvolve(img2, window, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = signal.fftconvolve(img1*img1, window, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(img2*img2, window, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(img1*img2, window, mode='valid') - mu1_mu2

    if cs_map:
        return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))/((mu1_sq + mu2_sq + C1) *
                    (sigma1_sq + sigma2_sq + C2)), 
                (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                    (sigma1_sq + sigma2_sq + C2))

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
