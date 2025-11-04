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
            'val_classification_predictions': list()
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
        This modified version plots:
        1. Total Train/Val Loss + Mean FG Dice (Original Plot)
        2. Seg/Cls Train/Val Losses
        3. Validation Metrics (Dice, Accuracy, Macro F1)
        4. Epoch Duration & Learning Rate
        """
        
        # --- Robust Epoch Calculation ---
        # We list all keys that are *supposed* to be logged every epoch.
        # This prevents un-logged keys (like 'val_classification_targets') 
        # from breaking the min() function and causing an empty plot.
        core_logging_keys = [
            'train_losses', 'val_losses', 'mean_fg_dice', 'ema_fg_dice', 'lrs',
            'epoch_start_timestamps', 'epoch_end_timestamps', 'train_loss_seg',
            'train_loss_cls', 'val_loss_seg', 'val_loss_cls', 'val_macro_f1', 'val_accuracy'
        ]

        # Get lengths of all core logs that are actually present and have entries
        logged_lengths = []
        for key in core_logging_keys:
            if key in self.my_fantastic_logging and len(self.my_fantastic_logging[key]) > 0:
                logged_lengths.append(len(self.my_fantastic_logging[key]))

        if not logged_lengths:
            # No logging data found, cannot plot.
            print("Warning: No logging data found to plot progress. Skipping plot generation.")
            return

        # Determine the number of epochs to plot
        epoch = min(logged_lengths) - 1  # lists of epoch 0 have len 1
        if epoch < 0:
            # Not enough data to plot (e.g., only epoch 0 started but didn't finish)
            print("Warning: Not enough logging data to plot progress. Skipping plot generation.")
            return

        x_values = list(range(epoch + 1))

        # --- Setup Figure ---
        sns.set(font_scale=2.5)
        # We now need 4 subplots
        fig, ax_all = plt.subplots(4, 1, figsize=(30, 68)) # Increased size for 4 plots

        # --- Plot 1: Total Loss and Pseudo Dice (Original) ---
        ax = ax_all[0]
        ax.set_title("Total Loss and Pseudo Dice")
        ax2 = ax.twinx()
        ax.plot(x_values, self.my_fantastic_logging['train_losses'][:epoch + 1], color='b', ls='-', label="loss_tr", linewidth=4)
        ax.plot(x_values, self.my_fantastic_logging['val_losses'][:epoch + 1], color='r', ls='-', label="loss_val", linewidth=4)
        ax2.plot(x_values, self.my_fantastic_logging['mean_fg_dice'][:epoch + 1], color='g', ls='dotted', label="pseudo dice",
                linewidth=3)
        ax2.plot(x_values, self.my_fantastic_logging['ema_fg_dice'][:epoch + 1], color='g', ls='-', label="pseudo dice (mov. avg.)",
                linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("Total Loss")
        ax2.set_ylabel("Pseudo Dice")
        ax.legend(loc=(0.0, 1.01)) # Adjusted location
        ax2.legend(loc=(0.25, 1.01)) # Adjusted location

        # --- Plot 2: Segmentation and Classification Losses ---
        ax = ax_all[1]
        ax.set_title("Component Losses (Seg vs. Cls)")
        ax2 = ax.twinx()
        
        # Plot Seg Losses
        ax.plot(x_values, self.my_fantastic_logging['train_loss_seg'][:epoch + 1], color='b', ls='-', label="train_loss_seg", linewidth=4)
        ax.plot(x_values, self.my_fantastic_logging['val_loss_seg'][:epoch + 1], color='r', ls='-', label="val_loss_seg", linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("Segmentation Loss")
        ax.legend(loc=(0.0, 1.01))

        # Plot Cls Losses
        ax2.plot(x_values, self.my_fantastic_logging['train_loss_cls'][:epoch + 1], color='b', ls='--', label="train_loss_cls", linewidth=4)
        ax2.plot(x_values, self.my_fantastic_logging['val_loss_cls'][:epoch + 1], color='r', ls='--', label="val_loss_cls", linewidth=4)
        ax2.set_ylabel("Classification Loss")
        ax2.legend(loc=(0.25, 1.01))

        # --- Plot 3: Validation Metrics (Dice, Acc, F1) ---
        ax = ax_all[2]
        ax.set_title("Validation Metrics")
        ax.plot(x_values, self.my_fantastic_logging['mean_fg_dice'][:epoch + 1], color='g', ls='-', label="Val Mean Dice", linewidth=4)
        ax.plot(x_values, self.my_fantastic_logging['val_accuracy'][:epoch + 1], color='b', ls='-', label="Val Accuracy", linewidth=4)
        ax.plot(x_values, self.my_fantastic_logging['val_macro_f1'][:epoch + 1], color='r', ls='-', label="Val Macro F1", linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("Metric")
        ax.legend(loc=(0.0, 1.01))
        ax.set_ylim(bottom=max(0, ax.get_ylim()[0]), top=min(1.05, ax.get_ylim()[1])) # Metrics are typically [0, 1]

        # --- Plot 4: Epoch Time and Learning Rate ---
        ax = ax_all[3]
        ax.set_title("System Stats")
        ax2 = ax.twinx()

        # Plot Epoch Time
        epoch_times = [i - j for i, j in zip(self.my_fantastic_logging['epoch_end_timestamps'][:epoch + 1],
                                            self.my_fantastic_logging['epoch_start_timestamps'][:epoch + 1])]
        ax.plot(x_values, epoch_times, color='b', ls='-', label="epoch duration", linewidth=4)
        ylim = [0] + [ax.get_ylim()[1]]
        ax.set(ylim=ylim)
        ax.set_xlabel("epoch")
        ax.set_ylabel("time [s]")
        
        # Plot Learning Rate
        ax2.plot(x_values, self.my_fantastic_logging['lrs'][:epoch + 1], color='r', ls='-', label="learning rate", linewidth=4)
        ax2.set_ylabel("learning rate")

        # Combine legends for the fourth plot
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=(0.0, 1.01))

        # --- Save Figure ---
        plt.tight_layout()
        fig.savefig(join(output_folder, "progress.png"))
        plt.close()

    def get_checkpoint(self):
        return self.my_fantastic_logging

    def load_checkpoint(self, checkpoint: dict):
        self.my_fantastic_logging = checkpoint
