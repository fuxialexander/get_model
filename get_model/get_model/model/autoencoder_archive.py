class GETFinetuneAuto(nn.Module):
    """A GET model for finetuning using autoencoder."""
    """Goal: [B,L,4] -> [B,L,M], then learning Latent from [B,L,M] and recover back to [B,L,4] (M stands for motif class)
        Architecture: Conv1D motif scanner + VAE + Conv1D motif to sequence decoder"""

    def __init__(
        self,
        num_motif=637,
        motif_dim=639,
        embed_dim=768,
        num_regions=200,
        motif_prior=True,
        num_layers=7,
        d_model=768,
        nhead=1,
        dropout=0.3,
        output_dim=1,
    ):
        super().__init__()
        self.num_regions = num_regions
        self.num_motif = num_motif
        self.motif_prior = motif_prior
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        self.output_dim = output_dim
        self.motif_scanner = MotifScanner(
            num_motif=num_motif, include_reverse_complement=True,
            bidirectional_except_ctcf=True
        )
        self.apply(self._init_weights)
        self.seq_len = 1024

        self.mu_encoder = nn.Sequential(
            # Projection from [B, 1, 256, 1024] -> [B, 64, 4, 16]
            nn.Conv2d(1, 2, kernel_size=(2, 2), stride=2),  # Output: [B, 2, 128, 512]
            nn.ReLU(),
            nn.Conv2d(2, 4, kernel_size=(2, 2), stride=2),  # Output: [B, 4, 64, 256]
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=(2, 2), stride=2),  # Output: [B, 8, 32, 128]
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=(2, 2), stride=2),  # Output: [B, 16, 16, 64]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(2, 2), stride=2),  # Output: [B, 32, 8, 32]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(2, 2), stride=2),  # Output: [B, 64, 4, 16]
            nn.ReLU(),
        )

        self.logvar_encoder = nn.Sequential(
            # Projection from [B, 1, 256, 1024] -> [B, 64, 4, 16]
            nn.Conv2d(1, 2, kernel_size=(2, 2), stride=2),  # Output: [B, 2, 128, 512]
            nn.ReLU(),
            nn.Conv2d(2, 4, kernel_size=(2, 2), stride=2),  # Output: [B, 4, 64, 256]
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=(2, 2), stride=2),  # Output: [B, 8, 32, 128]
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=(2, 2), stride=2),  # Output: [B, 16, 16, 64]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(2, 2), stride=2),  # Output: [B, 32, 8, 32]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(2, 2), stride=2),  # Output: [B, 64, 4, 16]
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            # [B, 64, 4, 16] -> [B, 1, 256, 1024]
            nn.ConvTranspose2d(64, 32, kernel_size=(2, 2), stride=2),  # Output: [B, 32, 8, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=(2, 2), stride=2),  # Output: [B, 16, 16, 64]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=(2, 2), stride=2),  # Output: [B, 8, 32, 128]
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=(2, 2), stride=2),  # Output: [B, 4, 64, 256]
            nn.ReLU(),
            nn.ConvTranspose2d(4, 2, kernel_size=(2, 2), stride=2),  # Output: [B, 2, 128, 512]
            nn.ReLU(),
            nn.ConvTranspose2d(2, 1, kernel_size=(2, 2), stride=2),  # Output: [B, 1, 256, 1024]
            nn.ReLU()
        )

        self.motif_proj = nn.Conv1d(motif_dim, 256, 1, bias=True)
        self.motif_proj_back = nn.Conv1d(in_channels=256, out_channels=639, kernel_size=1,bias=True)
        self.motifdecoder = motif2seqScanner(motif_dim, motif_dim//2, 4)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, peak_seq, atac, mask, padding_mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std, other_labels, hic_matrix, train_motif_only = False):
        """labels_data is (B, R, C), C=2 for expression. R=max_n_peaks"""
        #def forward(self, peak_seq, motif_mean_std):
        # peak_seq: [B, L, 4]
        # [B, L, 4] --> [B, L, 639]
        #print(f"peak size: f{peak_seq.shape}")
        x = self.motif_scanner(peak_seq)
        x = x - motif_mean_std[:,0, :].unsqueeze(1)
        x = x / motif_mean_std[:,1, :].unsqueeze(1)
        x = F.relu(x)
        motif_emb = x
        B, L, _ = x.shape
        self.seq_len = L
        #print(f"Input motif embedding shape is: {x.shape}")

        x = x.transpose(1, 2)  # Reshape for Conv1d: (B, L, M) to (B, M, L)
        x = self.motif_proj(x)
        x = x.unsqueeze(1)  # Add channel dimension: [B, 1, 256, 1024]

        mu = self.mu_encoder(x)
        logvar = self.logvar_encoder(x)
        latent_x = self.reparameterize(mu, logvar)
        #print(f"mu size: {mu.shape}")
        #print(f"logvar size: {logvar.shape}")
        #print(f"latent size: {latent_x.shape}")
        reformed_x = self.decoder(latent_x) # Decoder [B, 64, 4, 16] -> [B, 1, 256, 1024]]
        reformed_x = reformed_x.squeeze(1)  # [16, 1, 256, 1024] -> [16, 256, 1024]

        #latent_x = self.encoder(x) # Encoder [B, 1, 256, 1024]] -> [B, 64, 4, 16]
        mu = self.mu_encoder(x)
        logvar = self.logvar_encoder(x)
        latent_x = self.reparameterize(mu, logvar)
        #print(f"mu size: {mu.shape}")
        #print(f"logvar size: {logvar.shape}")
        #print(f"latent size: {latent_x.shape}")
        reformed_x = self.decoder(latent_x) # Decoder [B, 64, 4, 16] -> [B, 1, 256, 1024]]
        reformed_x = reformed_x.squeeze(1)  # [16, 1, 256, 1024] -> [16, 256, 1024]
        reformed_x = self.motif_proj_back(reformed_x) #[16, 256, 1024] -> [16, 639, 1024]
        #print(f"reconstructed data size: {reformed_x.transpose(1,2).shape}")
        if train_motif_only == True:
            return motif_emb, reformed_x.transpose(1,2), latent_x, mu, logvar
        else:
            output_x = self.motifdecoder(reformed_x.transpose(1,2)) # [16, 639, 1024] -> [16, 1024, 4]
            #print(f"output size: {output_x.shape}")
            if (output_x.shape != peak_seq.shape):
                print("Warning! input_peak_seq shape needs to equal to output_seq shape!")
            return peak_seq, output_x, motif_emb, reformed_x.transpose(1,2), latent_x, mu, logvar

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}