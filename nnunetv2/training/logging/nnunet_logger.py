import matplotlib
from batchgenerators.utilities.file_and_folder_operations import join

matplotlib.use('agg')
import seaborn as sns
import matplotlib.pyplot as plt


class nnUNetLogger(object):
    """
    This class is really trivial. Don't expect cool functionality here. This is my makeshift solution to problems
    arising from out-of-sync epoch numbers and numbers of logged loss values. It also simplifies the trainer class a
    little

    YOU MUST LOG EXACTLY ONE VALUE PER EPOCH FOR EACH OF THE LOGGING ITEMS! DONT FUCK IT UP
    """
    def __init__(self, verbose: bool = False):
        self.my_fantastic_logging = {
            'mean_fg_dice': list(),
            'ema_fg_dice': list(),
            'dice_per_class_or_region': list(),
            'train_losses': list(),
            'val_losses': list(),
            'lrs': list(),
            'epoch_start_timestamps': list(),
            'epoch_end_timestamps': list(),
            'train_loss_seg': list(),
            'train_loss_cls': list(),
            'val_loss_seg': list(),
            'val_loss_cls': list(),
            'val_macro_f1': list(),
            'val_accuracy': list(),
            'val_classification_targets': list(),
            'val_classification_predictions': list(),
            # Added for whole-pancreas metric & components
            'val_dice_whole_pancreas': list(),
            'val_wp_tp': list(),
            'val_wp_fp': list(),
            'val_wp_fn': list()
        }
        self.verbose = verbose
        # shut up, this logging is great

    def log(self, key, value, epoch: int):
        """
        sometimes shit gets messed up. We try to catch that here
        """
        assert key in self.my_fantastic_logging.keys() and isinstance(self.my_fantastic_logging[key], list), \
            'This function is only intended to log stuff to lists and to have one entry per epoch'

        if self.verbose: print(f'logging {key}: {value} for epoch {epoch}')

        if len(self.my_fantastic_logging[key]) < (epoch + 1):
            self.my_fantastic_logging[key].append(value)
        else:
            assert len(self.my_fantastic_logging[key]) == (epoch + 1), 'something went horribly wrong. My logging ' \
                                                                       'lists length is off by more than 1'
            print(f'maybe some logging issue!? logging {key} and {value}')
            self.my_fantastic_logging[key][epoch] = value

        # handle the ema_fg_dice special case! It is automatically logged when we add a new mean_fg_dice
        if key == 'mean_fg_dice':
            new_ema_pseudo_dice = self.my_fantastic_logging['ema_fg_dice'][epoch - 1] * 0.9 + 0.1 * value \
                if len(self.my_fantastic_logging['ema_fg_dice']) > 0 else value
            self.log('ema_fg_dice', new_ema_pseudo_dice, epoch)

    def plot_progress_png(self, output_folder):
        """
        Renders plots for training monitoring.
        如果某个 key 没有日志或长度不够，就不画这条线（或者只画已有的部分），
        不会再因为空 list / 长度不一致崩掉了。
        """
        # 这些 key 预期是“每个 epoch 都记一次”的核心指标
        core_logging_keys = [
            'train_losses', 'val_losses', 'mean_fg_dice', 'ema_fg_dice', 'lrs',
            'epoch_start_timestamps', 'epoch_end_timestamps',
            'train_loss_seg', 'train_loss_cls',
            'val_loss_seg', 'val_loss_cls',
            'val_macro_f1', 'val_accuracy'
        ]

        # 统计实际有数据的长度
        logged_lengths = []
        for key in core_logging_keys:
            vals = self.my_fantastic_logging.get(key, [])
            if isinstance(vals, list) and len(vals) > 0:
                logged_lengths.append(len(vals))

        if not logged_lengths:
            print("Warning: No logging data found to plot progress. Skipping plot generation.")
            return

        # 所有曲线最多画到的 epoch
        epoch = min(logged_lengths) - 1
        if epoch < 0:
            print("Warning: Not enough logging data to plot progress. Skipping plot generation.")
            return

        def get_xy(key):
            """
            安全地取出某个 key 对应的 x/y：
            - 如果没有数据，返回 (None, None) -> 不画；
            - 如果长度 < epoch+1，只画已有的那一部分。
            """
            vals = self.my_fantastic_logging.get(key, None)
            if vals is None or not isinstance(vals, list) or len(vals) == 0:
                return None, None
            max_len = min(len(vals), epoch + 1)
            x = list(range(max_len))
            y = vals[:max_len]
            return x, y

        import matplotlib
        matplotlib.use('agg')
        import seaborn as sns
        import matplotlib.pyplot as plt
        from batchgenerators.utilities.file_and_folder_operations import join

        sns.set(font_scale=2.5)
        fig, ax_all = plt.subplots(4, 1, figsize=(30, 68))

        # -------- Plot 1: Total Loss & Pseudo Dice --------
        ax = ax_all[0]
        ax.set_title("Total Loss and Pseudo Dice")
        ax2 = ax.twinx()

        x_tr, y_tr = get_xy('train_losses')
        x_val, y_val = get_xy('val_losses')
        x_dice, y_dice = get_xy('mean_fg_dice')
        x_dice_ema, y_dice_ema = get_xy('ema_fg_dice')

        if x_tr is not None:
            ax.plot(x_tr, y_tr, color='b', ls='-', label="loss_tr", linewidth=4)
        if x_val is not None:
            ax.plot(x_val, y_val, color='r', ls='-', label="loss_val", linewidth=4)
        if x_dice is not None:
            ax2.plot(x_dice, y_dice, color='g', ls='dotted', label="pseudo dice", linewidth=3)
        if x_dice_ema is not None:
            ax2.plot(x_dice_ema, y_dice_ema, color='g', ls='-', label="pseudo dice (mov. avg.)", linewidth=4)

        ax.set_xlabel("epoch")
        ax.set_ylabel("Total Loss")
        ax2.set_ylabel("Pseudo Dice")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc=(0.0, 1.01))
        if ax2.get_legend_handles_labels()[0]:
            ax2.legend(loc=(0.25, 1.01))

        # -------- Plot 2: Seg & Cls Losses --------
        ax = ax_all[1]
        ax.set_title("Component Losses (Seg vs. Cls)")
        ax2 = ax.twinx()

        x_tr_seg, y_tr_seg = get_xy('train_loss_seg')
        x_val_seg, y_val_seg = get_xy('val_loss_seg')
        x_tr_cls, y_tr_cls = get_xy('train_loss_cls')
        x_val_cls, y_val_cls = get_xy('val_loss_cls')

        if x_tr_seg is not None:
            ax.plot(x_tr_seg, y_tr_seg, color='b', ls='-', label="train_loss_seg", linewidth=4)
        if x_val_seg is not None:
            ax.plot(x_val_seg, y_val_seg, color='r', ls='-', label="val_loss_seg", linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("Segmentation Loss")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc=(0.0, 1.01))

        if x_tr_cls is not None:
            ax2.plot(x_tr_cls, y_tr_cls, color='b', ls='--', label="train_loss_cls", linewidth=4)
        if x_val_cls is not None:
            ax2.plot(x_val_cls, y_val_cls, color='r', ls='--', label="val_loss_cls", linewidth=4)
        ax2.set_ylabel("Classification Loss")
        if ax2.get_legend_handles_labels()[0]:
            ax2.legend(loc=(0.25, 1.01))

        # -------- Plot 3: Val Metrics (Dice / Acc / F1) --------
        ax = ax_all[2]
        ax.set_title("Validation Metrics")

        x_dice, y_dice = get_xy('mean_fg_dice')
        x_acc, y_acc = get_xy('val_accuracy')
        x_f1, y_f1 = get_xy('val_macro_f1')

        if x_dice is not None:
            ax.plot(x_dice, y_dice, color='g', ls='-', label="Val Mean Dice", linewidth=4)
        if x_acc is not None:
            ax.plot(x_acc, y_acc, color='b', ls='-', label="Val Accuracy", linewidth=4)
        if x_f1 is not None:
            ax.plot(x_f1, y_f1, color='r', ls='-', label="Val Macro F1", linewidth=4)

        ax.set_xlabel("epoch")
        ax.set_ylabel("Metric")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc=(0.0, 1.01))
        # metrics 一般 [0,1]，做个裁剪
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(bottom=max(0, ymin), top=min(1.05, ymax))

        # -------- Plot 4: Epoch Time & LR --------
        ax = ax_all[3]
        ax.set_title("System Stats")
        ax2 = ax.twinx()

        # epoch 时间
        start = self.my_fantastic_logging.get('epoch_start_timestamps', [])
        end = self.my_fantastic_logging.get('epoch_end_timestamps', [])
        if len(start) > 0 and len(end) > 0:
            max_len_t = min(len(start), len(end), epoch + 1)
            x_t = list(range(max_len_t))
            epoch_times = [e - s for e, s in zip(end[:max_len_t], start[:max_len_t])]
            ax.plot(x_t, epoch_times, color='b', ls='-', label="epoch duration", linewidth=4)
            ylim = [0, ax.get_ylim()[1]]
            ax.set(ylim=ylim)
        ax.set_xlabel("epoch")
        ax.set_ylabel("time [s]")

        # 学习率
        x_lr, y_lr = get_xy('lrs')
        if x_lr is not None:
            ax2.plot(x_lr, y_lr, color='r', ls='-', label="learning rate", linewidth=4)
        ax2.set_ylabel("learning rate")

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        if lines or lines2:
            ax2.legend(lines + lines2, labels + labels2, loc=(0.0, 1.01))

        plt.tight_layout()
        fig.savefig(join(output_folder, "progress.png"))
        plt.close()


    def get_checkpoint(self):
        return self.my_fantastic_logging

    def load_checkpoint(self, checkpoint: dict):
        self.my_fantastic_logging = checkpoint
