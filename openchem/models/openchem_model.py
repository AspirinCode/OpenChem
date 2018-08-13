def fit(model, scheduler, train_loader, optimizer, criterion, params,
        eval=False, val_loader=None):
    cur_epoch = 0
    logdir = params['logdir']
    print_every = params['print_every']
    save_every = params['save_every']
    n_epochs = params['num_epochs']
    logger = Logger(logdir + '/tensorboard_log/')
    start = time.time()
    loss_total = 0
    n_batches = 0
    if scheduler is not None:
        scheduler = scheduler.scheduler
    all_losses = []
    val_losses = []
    max_metrics = -1000

    for epoch in range(cur_epoch, n_epochs + cur_epoch):
        for i_batch, sample_batched in enumerate(train_loader):
            batch_input, batch_target = model.module.cast_inputs(sample_batched)
            loss = train_step(model, optimizer, criterion,
                              batch_input, batch_target)
            if model.module.world_size > 1:
                reduced_loss = reduce_tensor(loss, model.module.world_size)
            else:
                reduced_loss = loss.clone()
            loss_total += reduced_loss.item()
            n_batches += 1
        cur_loss = loss_total / n_batches
        all_losses.append(cur_loss)

        if epoch % print_every == 0:
            if print_logs(model.module.world_size):
                print('TRAINING: [Time: %s, Epoch: %d, Progress: %d%%, '
                      'Loss: %.4f]' % (time_since(start), epoch,
                                       epoch / n_epochs * 100, cur_loss))
            if eval:
                assert val_loader is not None
                val_loss, metrics = evaluate(model, val_loader, criterion)
                val_losses.append(val_loss)
                info = {'Train loss': cur_loss, 'Validation loss': val_loss,
                        'Validation metrics': metrics,
                        'LR': optimizer.param_groups[0]['lr']}
            else:
                info = {'Train loss': cur_loss,
                        'LR': optimizer.param_groups[0]['lr']}

            if print_logs(model.module.world_size):
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, epoch + 1)

                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    try:
                        logger.histo_summary(tag, value.detach().cpu().numpy(),
                                         epoch + 1)
                        logger.histo_summary(tag + '/grad',
                                         value.grad.detach().cpu().numpy(),
                                         epoch + 1)
                    except:
                        pass

        if epoch % save_every == 0 and print_logs(model.module.world_size):
            torch.save(model.state_dict(), logdir + '/checkpoint/epoch_' + str(epoch))
        elif eval and metrics > max_metrics:
            max_metrics = metrics
            torch.save(model.state_dict(), logdir + '/checkpoint/best')

        loss_total = 0
        n_batches = 0
        if scheduler is not None:
            scheduler.step()

    return all_losses, val_losses
