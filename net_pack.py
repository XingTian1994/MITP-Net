from functions import *

class Encoder(nn.Module):
    def __init__(self, in_dim=30, f_indim=18, f_hidim=128, gru_in_dim=12, hidden_dim=128, out_dim=12, cuda='cuda:0'):
        super().__init__()
        self.active = nn.Tanh()
        self.cuda = cuda
        # fusion net
        self.f_layer1 = nn.Linear(f_indim, f_hidim)
        self.f_ln1 = nn.LayerNorm(f_hidim, elementwise_affine=True)
        self.f_layer2 = nn.Linear(f_hidim, f_hidim)
        self.f_ln2 = nn.LayerNorm(f_hidim, elementwise_affine=True)
        self.f_layer3 = nn.Linear(f_hidim, f_hidim)
        self.f_ln3 = nn.LayerNorm(f_hidim, elementwise_affine=True)
        self.f_layer4 = nn.Linear(f_hidim, gru_in_dim)
        # GRU net
        self.pre_linear = nn.Linear(gru_in_dim, hidden_dim)
        self.pre_linear2 = nn.Linear(in_dim, hidden_dim)
        self.pre_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.ln = nn.LayerNorm(hidden_dim, elementwise_affine=True)
        self.gru1 = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden_dim, elementwise_affine=True)
        self.gru2 = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.ln2 = nn.LayerNorm(hidden_dim, elementwise_affine=True)
        self.gru2 = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.ln2 = nn.LayerNorm(hidden_dim, elementwise_affine=True)
        self.gru3 = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.ln3 = nn.LayerNorm(hidden_dim, elementwise_affine=True)
        self.gru4 = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.ln4 = nn.LayerNorm(hidden_dim, elementwise_affine=True)
        self.gru5 = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, fusion, mode):
        if fusion:
            # devide data
            if mode == 'train':
                indoor_t = torch.zeros(len(x), len(x[0]), 0).to(device=self.cuda)
                envir = torch.zeros(len(x), len(x[0]), 0).to(device=self.cuda)
            elif mode == 'test':
                indoor_t = torch.zeros(len(x), len(x[0]), 0)
                envir = torch.zeros(len(x), len(x[0]), 0)
            for i in range(6):
                temp_t = x[:, :, (5 * i + 2):(5 * i + 4)]
                temp_envir = torch.cat((x[:, :, (5 * i):(5 * i + 2)], x[:, :, (5 * i + 4)].unsqueeze(2)), 2)
                indoor_t = torch.cat((indoor_t, temp_t), 2)
                envir = torch.cat((envir, temp_envir), 2)

            # env net
            envir = self.f_layer1(envir)
            envir = self.f_ln1(envir)
            envir = self.active(envir)
            envir = self.f_layer2(envir)
            envir = self.f_ln2(envir)
            envir = self.active(envir)
            envir = self.f_layer3(envir)
            envir = self.f_ln3(envir)
            envir = self.active(envir)
            envir = self.f_layer4(envir)

            # fusion
            if fusion == 'Hadamard':
                g_in = indoor_t * envir
            elif fusion == 'add':
                g_in = indoor_t + envir
            out = self.pre_linear(g_in)
        else:
            out = self.pre_linear2(x)
        out = self.active(out)
        out, _1 = self.pre_gru(out)
        out = self.ln(out)
        out, _2 = self.gru1(out)
        out = self.ln1(out)
        out, _3 = self.gru2(out)
        out = self.ln2(out)
        out, _4 = self.gru3(out)
        out = self.ln3(out)
        out, _5 = self.gru4(out)
        out = self.ln4(out)
        out, _6 = self.gru5(out)
        out = self.linear(out)
        out = self.active(out)
        hidden = torch.cat((_1,_2,_3,_4,_5,_6), 0)  # [layers, batch size, hidden size]
        # hidden.transpose(0, 1)                       # [batch size, layers, hidden size]
        return hidden


class Decoder(nn.Module):
    def __init__(self, in_dim=12, n_layers=6, hidden_dim=128, out_dim=12):
        super().__init__()
        self.output_size = out_dim
        self.hidden_size = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(in_dim, hidden_dim, n_layers)
        self.linear = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, hidden):
        out, de_hidden = self.gru(x, hidden)
        out = self.linear(out)
        return out, de_hidden

class seq2seq(nn.Module):
    def __init__(self, encoder, decoder, cuda='cuda:0'):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.cuda = cuda

    def forward(self, en, tr_y, pre_len, fusion_style, t_mode='train'):                        # en - input tensor
        batch_size = en.shape[0]
        outputs = torch.zeros(pre_len, batch_size, 12).to(self.cuda)
        # hid = self.encoder(en, fusion=fusion_style, mode='train')
        if t_mode == 'train':
            hid = self.encoder(en, fusion=fusion_style, mode='train')
        elif t_mode == 'test':
            hid = self.encoder(en, fusion=fusion_style, mode='test')
        de_input = torch.zeros(batch_size, 1, 12).to(self.cuda)
        for a in range(6):
            de_input[:, :, 2*a:2*a+2] = en[:, -1, 5*a+2:5*a+4].unsqueeze(1)
        de_input=de_input.permute(1,0,2)

        for p in range(pre_len):
            if t_mode == 'test':
                output, hid = self.decoder(de_input, hid)
            elif t_mode == 'train':
                if p == 0:
                    output, hid = self.decoder(de_input, hid)
                else:
                    a = tr_y.permute(1,0,2)
                    b = a[p-1].unsqueeze(0)
                    output, hid = self.decoder(b, hid)
            de_input = output
            outputs[p] = output.squeeze(0)
        outputs = outputs.permute(1, 0, 2)

        return outputs
