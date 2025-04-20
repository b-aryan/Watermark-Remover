import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import config
from dataset import WatermarkPatchDataset
from generator_model import Generator
from discriminator_model import Discriminator
from vgg_loss import VGGLoss
from utils import save_checkpoint, load_checkpoint


def train_fn(disc, gen, loader, opt_disc, opt_gen, l1_loss, vgg_loss, bce, g_scaler, d_scaler):
    loop = tqdm(loader, leave=True)
    last_D_loss = 0.0
    last_G_total = 0.0

    for idx, (marked_patches, clean_patches) in enumerate(loop):
        marked = marked_patches.to(config.DEVICE)
        clean = clean_patches.to(config.DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            fake_clean = gen(marked)
            D_real = disc(marked, clean)
            D_fake = disc(marked, fake_clean.detach())

            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        if config.CLIP_GRAD is not None:  # Added gradient clipping
            torch.nn.utils.clip_grad_norm_(disc.parameters(), config.CLIP_GRAD)
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generator
        with torch.cuda.amp.autocast():
            fake_clean = gen(marked)
            D_fake = disc(marked, fake_clean)

            G_gan_loss = bce(D_fake, torch.ones_like(D_fake)) * config.GAN_LOSS_LAMBDA
            G_l1 = l1_loss(fake_clean, clean) * config.L1_LAMBDA
            G_vgg = vgg_loss(fake_clean, clean) * config.VGG_LAMBDA
            G_total = G_gan_loss + G_l1 + G_vgg

        opt_gen.zero_grad()
        g_scaler.scale(G_total).backward()
        if config.CLIP_GRAD is not None:
            torch.nn.utils.clip_grad_norm_(gen.parameters(), config.CLIP_GRAD)
        g_scaler.step(opt_gen)
        g_scaler.update()

        # Store last losses
        last_D_loss = D_loss.item()
        last_G_total = G_total.item()

        # Update progress bar
        loop.set_postfix(
            D_loss=last_D_loss,
            G_total=last_G_total,
            G_l1=G_l1.item(),
            G_vgg=G_vgg.item()
        )

    return last_D_loss, last_G_total  # Return values for schedulers


def main():
    # Initialize models
    gen = Generator().to(config.DEVICE)
    disc = Discriminator().to(config.DEVICE)

    # Optimizers
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    # Added learning rate schedulers
    scheduler_gen = optim.lr_scheduler.ReduceLROnPlateau(opt_gen, patience=2)
    scheduler_disc = optim.lr_scheduler.ReduceLROnPlateau(opt_disc, patience=2)

    # Loss functions
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()
    VGG_LOSS = VGGLoss().to(config.DEVICE)

    # Load checkpoints
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE)

    # Dataset and dataloader
    train_dataset = WatermarkPatchDataset(
        mark_dir=config.MARK_DIR,
        nomark_dir=config.NOMARK_DIR,
        kernel_size=config.PATCH_KERNEL_SIZE,
        stride=config.PATCH_STRIDE,
        transform_input=config.transform_input,
        transform_target=config.transform_target
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    # Scalers for AMP
    g_scaler = torch.cuda.amp.GradScaler(enabled=config.USE_AMP)
    d_scaler = torch.cuda.amp.GradScaler(enabled=config.USE_AMP)

    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch [{epoch + 1}/{config.NUM_EPOCHS}]")
        D_loss, G_total = train_fn(disc, gen, train_loader, opt_disc, opt_gen,
                                  L1_LOSS, VGG_LOSS, BCE, g_scaler, d_scaler)

        # Update learning rate schedulers
        scheduler_gen.step(G_total)
        scheduler_disc.step(D_loss)

        # Save model checkpoints
        if config.SAVE_MODEL and (epoch + 1) % config.CHECKPOINT_SAVE_INTERVAL == 0:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

