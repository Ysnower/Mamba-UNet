from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from models.UltraLight_VM_UNet import UltraLight_VM_UNet
from engine import *
from utils import *
from configs.config_setting import setting_config
from dataloader import MydataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # "0, 1, 2, 3"


def main(config):

    print('#----------Creating logger----------#')
    if not os.path.exists(config.modelsSavePath):
        os.makedirs(config.modelsSavePath)
    log_dir = config.log_dir
    resume_model = config
    global logger
    logger = get_logger('train', log_dir)
    log_config_info(config, logger)
    print('#----------GPU init----------#')
    set_seed(config.seed)
    gpu_ids = [0]  # [0, 1, 2, 3]
    torch.cuda.empty_cache()
    print('#----------Preparing dataset----------#')

    train_dataset = MydataLoader(dataRoot="datasets/train/images", loadSize=config.load_size, training=True)
    val_dataset = MydataLoader(dataRoot="datasets/val/images", loadSize=config.load_size, training=False)
    train_loader = DataLoader(train_dataset,
                                batch_size=config.batch_size, 
                                shuffle=True,
                                pin_memory=True,
                                num_workers=config.num_workers,
                                drop_last=True)
    val_loader = DataLoader(val_dataset,
                                batch_size=config.batch_size,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers,
                                drop_last=True)
    print('#----------Prepareing Models----------#')
    model_cfg = config.model_config
    model = UltraLight_VM_UNet(num_classes=model_cfg['num_classes'], 
                               input_channels=model_cfg['input_channels'], 
                               c_list=model_cfg['c_list'], 
                               split_att=model_cfg['split_att'], 
                               bridge=model_cfg['bridge'],)
    model = model.cuda()
    numOfGPUS = 0  # single gpu training
    if numOfGPUS > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])
    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    scaler = GradScaler()
    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1
    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        if numOfGPUS > 1:
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        logger.info(log_info)
    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):

        # torch.cuda.empty_cache()

        train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            logger,
            config,
            scaler=scaler
        )
        if epoch % config.val_interval == 0:

            loss = val_one_epoch(
                    val_loader,
                    model,
                    criterion,
                    epoch,
                    logger,
                    config
                )
            if loss < min_loss:
                if numOfGPUS > 1:
                    torch.save(model.module.state_dict(), config.modelsSavePath+"/MambaUnet_best.pth")
                else:
                    torch.save(model.state_dict(), config.modelsSavePath+"/MambaUnet_best.pth")
                min_loss = loss
                min_epoch = epoch

            if numOfGPUS > 1:
                torch.save(
                    {
                        'epoch': epoch,
                        'min_loss': min_loss,
                        'min_epoch': min_epoch,
                        'loss': loss,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                    }, config.modelsSavePath+"/MambaUnet_latest{}.pth".format(epoch))
            else:
                torch.save(
                    {
                        'epoch': epoch,
                        'min_loss': min_loss,
                        'min_epoch': min_epoch,
                        'loss': loss,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                    }, config.modelsSavePath+"/MambaUnet_latest{}.pth".format(epoch))


if __name__ == '__main__':
    config = setting_config
    main(config)