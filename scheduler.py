from torch.optim.lr_scheduler import LambdaLR


def create_scheduler(args, optimizer):
    if 'num_training_steps' not in args:
        args['num_training_steps'] = args['epochs'] * args['step_per_epoch']
    print("### num_training_steps, ", args['num_training_steps'], flush=True)

    if isinstance(args['num_warmup_steps'], float):
        assert 0 <= args['num_warmup_steps'] < 1
        args['num_warmup_steps'] = int(args['num_training_steps'] * args['num_warmup_steps'])
    print("### num_warmup_steps, ", args['num_warmup_steps'], flush=True)

    print('sched:', args.sched, flush=True)

    if args.sched == 'linear':
        def lr_lambda(current_step: int):
            if current_step < args.num_warmup_steps:
                return float(current_step) / float(max(1, args.num_warmup_steps))
            return max(
                0.0, float(args.num_training_steps - current_step) / float(
                    max(1, args.num_training_steps - args.num_warmup_steps))
            )

        lr_scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=-1)

    elif args.sched == 'step':
        def lr_lambda(current_step: int):
            if current_step < (args.num_warmup_steps - args.num_warmup_steps / 3):
                return float(current_step) / float(max(1, args.num_warmup_steps))
            #baseline
            # elif current_step < args.num_warmup_steps * 4:
            #     tt = 1
            # elif current_step < args.num_warmup_steps * 7:
            #     tt = 0.5
            # else:
            #     tt = 0.2
            

            # rstp
            # elif current_step < ((args.num_warmup_steps - args.num_warmup_steps / 3) * 2):
            #     tt = 0.5
            # elif current_step < ((args.num_warmup_steps - args.num_warmup_steps / 3) * 3):
            #     tt = 0.3
            # elif current_step < (args.num_warmup_steps * 4):
            #     tt = 0.25
            # elif current_step < (args.num_warmup_steps * 9):
            #     tt = 0.2
            # else:
            #     tt = 0.1

            
            # icfg
            elif current_step < (args.num_warmup_steps * 3 - args.num_warmup_steps / 2) :
                tt = 1
            elif current_step < (args.num_warmup_steps * 4 - args.num_warmup_steps / 2):
                tt = 0.8
            elif current_step < (args.num_warmup_steps * 5) :
                tt = 0.4
            elif current_step < (args.num_warmup_steps * 7 - args.num_warmup_steps / 2):
                tt = 0.2
            else:
                tt = 0.1
            # cuhk
            # elif current_step < (args.num_warmup_steps - args.num_warmup_steps / 3) :
            #     tt = 1
            # elif current_step < (args.num_warmup_steps):
            #     tt = 0.5
            # elif current_step < (args.num_warmup_steps + args.num_warmup_steps / 3) :
            #     tt = 0.3
            # elif current_step < (args.num_warmup_steps * 2 + args.num_warmup_steps / 3):
            #     tt = 0.2
            # elif current_step < (args.num_warmup_steps * 4 + args.num_warmup_steps / 3):
            #     tt = 0.1
            # elif current_step < ((args.num_warmup_steps * 6)):
            #     tt = 0.05
            # elif current_step < ((args.num_warmup_steps * 8)):
            #     tt = 0.03
            # else:
            #     tt = 0.01

            return tt * max(
                0.0, float(args.num_training_steps - current_step) / float(
                    max(1, args.num_training_steps - args.num_warmup_steps))
            )

        lr_scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=-1)

    else:
        raise NotImplementedError(f"args.sched == {args.sched}")

    return lr_scheduler
