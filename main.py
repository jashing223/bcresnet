# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import os
from argparse import ArgumentParser
import shutil
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import wandb
from thop import profile
import json
from torchvision.ops import sigmoid_focal_loss
import umap
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import gradio as gr
import random
import warnings

from bcresnet import BCResNets
from utils import DownloadDataset, Padding, Preprocess, SpeechCommand, SplitDataset

warnings.simplefilter('ignore', UserWarning)

class Trainer:
    def __init__(self):
        """
        Constructor for the Trainer class.

        Initializes the trainer object with default values for the hyperparameters and data loaders.
        """

        parser = ArgumentParser()
        parser.add_argument("--ver", default=1, help="google speech command set version 1 or 2", type=int)
        parser.add_argument("--num_classes", default=12, help="number of classes", type=int)
        parser.add_argument("--tau", default=1, help="model size", type=float, choices=[1, 1.5, 2, 3, 6, 8])
        parser.add_argument("--lambda1", default=1, help="weight of keyword branch", type=float)
        parser.add_argument("--lambda2", default=1, help="weight of speech branch", type=float)
        parser.add_argument("--gpu", default=0, help="gpu device id", type=int)
        parser.add_argument("--download", help="download data", action="store_true")
        parser.add_argument("--eval", help="Only run evaluation", action="store_true")
        parser.add_argument("--plot", help="Only run umap plot", action="store_true")
        parser.add_argument("--demo", help="Only run demo", action="store_true")
        parser.add_argument("--ckpt", help="Path to checkpoint file for evaluation", type=str, default="")
        args = parser.parse_args()
        self.__dict__.update(vars(args))
        self.device = torch.device("cuda:%d" % self.gpu if torch.cuda.is_available() else "cpu")
        self._load_data()
        self._load_model()

        font_path = '/share/nas169/jethrowang/fonts/Times_New_Roman.ttf'
        font_prop = FontProperties(fname=font_path, size=17)

        # Add a list to track top 3 validation accuracies
        self.top_3_valid_accs = []
        
        # Create a directory to save checkpoints if it doesn't exist
        self.checkpoint_dir = f"./checkpoints/pavd_sr_tau_{self.tau}_ver_{self.ver}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if self.eval and not self.ckpt:
            raise ValueError("Please provide a checkpoint file using --ckpt <path> when using --eval mode.")

    def __call__(self):
        """
        Method that allows the object to be called like a function.

        Trains the model and presents the train/test progress.
        """

        wandb.init(entity="jashing223-national-taiwan-normal-university", project="pkws", name=f'pvad_sr_tau_{self.tau}_ver_{self.ver}')

        # train hyperparameters
        total_epoch = 100
        warmup_epoch = 5
        init_lr = 1e-1
        lr_lower_limit = 0

        # optimizer
        optimizer = torch.optim.SGD([
            {'params': list(self.model.cnn_head.parameters()) + list(self.model.BCBlocks.parameters()), 'weight_decay': 1e-3, 'momentum': 0.9},
            {'params': self.model.classifier1.parameters(), 'weight_decay': 1e-3, 'momentum': 0.9},
            {'params': self.model.classifier2.parameters(), 'weight_decay': 1e-3, 'momentum': 0.9},
            {'params': self.model.classifier3.parameters(), 'weight_decay': 1e-3, 'momentum': 0.9}
        ], lr=0)
        
        n_step_warmup = len(self.train_loader) * warmup_epoch
        total_iter = len(self.train_loader) * total_epoch
        iterations = 0

        # Best model tracking
        best_valid_acc = 0

        # train
        for epoch in range(total_epoch):
            self.model.train()
            for sample in tqdm(self.train_loader, desc="epoch %d, iters" % (epoch + 1)):
                # lr cos schedule
                iterations += 1
                if iterations < n_step_warmup:
                    lr = init_lr * iterations / n_step_warmup
                else:
                    lr = lr_lower_limit + 0.5 * (init_lr - lr_lower_limit) * (
                        1
                        + np.cos(
                            np.pi * (iterations - n_step_warmup) / (total_iter - n_step_warmup)
                        )
                    )
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                # Extract inputs, speaker embeddings, and labels
                inputs, speaker_embeddings, labels, speaker_labels = sample
                inputs = inputs.to(self.device)
                speaker_embeddings = speaker_embeddings.to(self.device)
                labels = labels.to(self.device)
                speaker_labels = speaker_labels.to(self.device)

                # Define multi-level labels
                speech_labels = (labels != 0).long().float()  # 0 -> non-speech, 1~11 -> speech
                # print(f'speech_labels: {speech_labels.shape}, {speech_labels}')
                speech_alpha = 1 - (speech_labels.sum().item() / len(speech_labels))  # positive ratio in speech_labels
                # speech_alpha = max(0.1, min(speech_alpha, 0.9))
                # print(f'speech_alpha: {speech_alpha}')

                # Only process keyword labels when speech samples exist
                if (labels >= 1).sum() > 0:
                    keyword_labels = (labels[labels >= 1] >= 2).long().float()  # 1 -> non-keyword, 2~11 -> keyword
                    keyword_alpha = 1 - (keyword_labels.sum().item() / len(keyword_labels))  # positive ratio in keyword_labels
                    # keyword_alpha = max(0.1, min(keyword_alpha, 0.9))
                else:
                    keyword_alpha = 0
                # print(f'keyword_labels: {keyword_labels.shape}, {keyword_labels}')
                # print(f'keyword_alpha: {keyword_alpha}')

                # Only process keyword class labels when keyword samples exist
                if (labels >= 2).sum() > 0:
                    keyword_class_labels = labels[labels >= 2] - 2
                # print(f'keyword_class_labels: {keyword_class_labels.shape}, {keyword_class_labels}')

                # Preprocess inputs
                inputs = self.preprocess_train(inputs, labels, augment=True)
                # print(f'processed_inputs: {inputs.shape}')

                # Get embeddings
                embeddings = self.model.encode(inputs)
                # print(f'all_embeddings: {embeddings.shape}')
                
                # Get all outputs with speaker conditioning
                speaker_outputs, keyword_outputs_raw, keyword_class_outputs_raw = self.model(inputs, speaker_embeddings)
                
                # Calculate speaker classification loss (3-class: same, different, silence) weighted pairwise loss
                speech_loss = self.wpl_loss(speaker_outputs, speaker_labels)

                # Initialize keyword loss and keyword class loss
                keyword_loss = torch.tensor(0.0, device=self.device)
                softmax_loss = torch.tensor(0.0, device=self.device)

                # Only process keyword/non-keyword classification when speech samples exist
                if (labels >= 1).sum() > 0:
                    keyword_outputs = keyword_outputs_raw[labels >= 1]
                    
                    keyword_loss = sigmoid_focal_loss(
                        inputs=keyword_outputs, 
                        targets=keyword_labels.unsqueeze(1), 
                        alpha=keyword_alpha, 
                        reduction='mean'
                    )

                    # Only process keyword class classification when keyword samples exist
                    if (labels >= 2).sum() > 0:
                        keyword_class_outputs = keyword_class_outputs_raw[labels >= 2]
                        
                        softmax_loss = F.cross_entropy(
                            keyword_class_outputs, 
                            keyword_class_labels
                        )

                # Calculate total loss
                loss = softmax_loss + self.lambda1 * keyword_loss + self.lambda2 * speech_loss
                wandb.log({"Total Loss": loss.item(), "Softmax Loss": softmax_loss.item(), "Keyword Loss": keyword_loss.item(), "Speech Loss": speech_loss.item(), "LR": lr})

                # Backpropagation and weight update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # valid
            print("cur lr check ... %.4f" % lr)
            # wandb.log({"LR": lr})
            with torch.no_grad():
                self.model.eval()
                valid_acc, valid_auroc, valid_f1, valid_fa, valid_eer = self.Test(self.valid_dataset, self.valid_loader, augment=True)
                print(f"Valid - Acc: {valid_acc:.3f}, AUROC: {valid_auroc:.3f}, F1: {valid_f1:.3f}, FA: {valid_fa:.3f}, EER: {valid_eer:.3f}")
                wandb.log({
                    "Epoch": epoch + 1,
                    "Valid_Acc": valid_acc,
                    "Valid_AUROC": valid_auroc,
                    "Valid_F1": valid_f1,
                    "Valid_FA": valid_fa,
                    "Valid_EER": valid_eer
                })

                # Save checkpoint for top 3 validation accuracies
                self._save_top_3_checkpoints(epoch, valid_acc)

        test_acc, test_auroc, test_f1, test_fa, test_eer = self.Test(self.test_dataset, self.test_loader, augment=False)  # official testset
        print(f"Last ckpt test - Acc: {test_acc:.3f}, AUROC: {test_auroc:.3f}, F1: {test_f1:.3f}, FA: {test_fa:.3f}, EER: {test_eer:.3f}")

        # After training, test the best checkpoint
        self._test_best_checkpoint()

        wandb.finish()

        print("End.")

    def _binary_eer(self,y_true, y_score):
        """計算 binary 的 EER"""
        fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
        return brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    def eer_score(self,y_true, y_score):
        """
        計算 Equal Error Rate (EER, %) for binary or multi-class (OVR only).
        
        參數:
        - y_true: 一維 array-like, 標籤
        - y_score: binary -> 一維預測分數
                multi-class -> shape (n_samples, n_classes)，每類的分數

        回傳:
        - eer (百分比, float 或 list)
        """
        y_true = np.array(y_true)
        y_score = np.array(y_score)

        # Binary
        if y_score.ndim == 1 or y_score.shape[1] == 1:
            return self._binary_eer(y_true, y_score.ravel()) * 100.0

        # Multi-class OVR
        classes = np.unique(y_true)
        y_true_bin = label_binarize(y_true, classes=classes)

        eer_list = []
        for i in range(len(classes)):
            eer = self._binary_eer(y_true_bin[:, i], y_score[:, i])
            eer_list.append(eer * 100.0)

        return np.mean(eer_list)

    def Test(self, dataset, loader, augment):
        """
        Tests the model on a given dataset and calculates accuracy, AUROC, F1-score, and false alarm rate.

        Parameters:
            dataset (Dataset): The dataset to test the model on.
            loader (DataLoader): The data loader to use for batching the data.
            augment (bool): Flag indicating whether to use data augmentation during testing.

        Returns:
            float: The accuracy of the model on the given dataset.
            float: The AUROC score for the multi-class classification task.
            float: The F1-score for the multi-class classification task.
            float: The false alarm rate (FA), where label 0 or 1 is misclassified as label 2~11.
        """

        self.model.eval()

        all_labels = []
        all_outputs = []  # probabilities
        all_predictions = []

        true_count = 0.0
        num_testdata = float(len(dataset))
        fa_count = 0
        neg_total = 0
        confusion_mat = np.zeros((self.num_classes, self.num_classes))

        for sample in loader:
            inputs, speaker_embeddings, raw_labels, speaker_labels = sample
            speaker_embeddings = speaker_embeddings.to(self.device)
            inputs = inputs.to(self.device)
            speaker_labels = speaker_labels.to(self.device)
            raw_labels = raw_labels.to(self.device)

            # print(f'raw_labels: {raw_labels}')
            inputs = self.preprocess_test(inputs, labels=raw_labels, is_train=False, augment=augment)
            # Use the first speaker embedding for inference (batch size 1 for test)
            outputs = self.model.inference(inputs, speaker_embeddings)
            # print(f'outputs: {outputs}')
            condition_mask = speaker_labels != 2
            labels = torch.where(condition_mask, torch.tensor(0), raw_labels)
            # print(f"speaker_labels: {speaker_labels}, condition_mask: {condition_mask}")
            # print(f'labels: {labels}')
            # Collect all predictions and labels
            predictions = torch.argmax(outputs, dim=-1)
            # print(f'predictions: {predictions}')
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.cpu().detach().numpy())  # probabilities
            all_predictions.extend(predictions.cpu().numpy())

            # Update confusion matrix
            batch_confusion = confusion_matrix(labels.cpu().numpy(), predictions.cpu().numpy(), labels=np.arange(self.num_classes))
            confusion_mat += batch_confusion            

            # Accuracy calculation
            true_count += torch.sum(predictions == labels).detach().cpu().numpy()
        acc = true_count / num_testdata * 100.0  # percentage

        # print(f'all_labels: {all_labels}')
        # print(f'all_outputs: {all_outputs}')
        # print(f'all_predictions: {all_predictions}')

        # AUROC calculation
        if len(set(all_labels)) < self.num_classes:
            auroc = float('nan')
        else:
            auroc = roc_auc_score(np.array(all_labels), np.array(all_outputs), average='macro', multi_class='ovr') * 100.0
        
        # calc eer score
        eer = self.eer_score(all_labels, all_outputs)

        # F1-score calculation
        f1 = f1_score(np.array(all_labels), np.array(all_predictions), average='macro') * 100.0
        
        # False alarm rate calculation
        for i in [0, 1]:  # Only consider label 0 (_silence_) and label 1 (_unknown_)
            fa_count += np.sum(confusion_mat[i, 2:])  # Count misclassifications to 2~11
            neg_total += np.sum(confusion_mat[i, :])   # Total occurrences of class 0 or 1
        if neg_total == 0:
            fa = None
        else:
            fa = fa_count / neg_total * 100.0

        return acc, auroc, f1, fa, eer

    def _load_data(self):
        """
        Private method that loads data into the object.

        Downloads and splits the data if necessary.
        """

        print("Check google speech commands dataset v1 or v2 ...")
        if not os.path.isdir("/home/jashing223/datasets/GSC"):
            os.mkdir("/home/jashing223/datasets/GSC")
        base_dir = "/home/jashing223/datasets/GSC/speech_commands_v0.01"
        url = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.01.tar.gz"
        url_test = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_test_set_v0.01.tar.gz"
        if self.ver == 2:
            base_dir = base_dir.replace("v0.01", "v0.02")
            url = url.replace("v0.01", "v0.02")
            url_test = url_test.replace("v0.01", "v0.02")
        test_dir = base_dir.replace("commands", "commands_test_set")
        if self.download:
            old_dirs = glob(base_dir.replace("commands_", "commands_*"))
            for old_dir in old_dirs:
                shutil.rmtree(old_dir)
            os.mkdir(test_dir)
            DownloadDataset(test_dir, url_test)
            os.mkdir(base_dir)
            DownloadDataset(base_dir, url)
            SplitDataset(base_dir)
            print("Done...")

        # Define data loaders
        train_dir = "%s/train_12class" % base_dir
        valid_dir = "%s/valid_12class" % base_dir
        noise_dir = "%s/_background_noise_" % base_dir

        # Paths to speaker embeddings
        train_embeddings_path = os.path.join(os.path.dirname(__file__), "train_embeddings.pt")
        valid_embeddings_path = os.path.join(os.path.dirname(__file__), "valid_embeddings.pt")
        test_embeddings_path = os.path.join(os.path.dirname(__file__), "test_embeddings.pt")

        transform = transforms.Compose([Padding()])
        self.train_dataset = SpeechCommand(train_dir, self.ver, transform=transform, 
                                          embeddings_path=train_embeddings_path)
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=100, shuffle=True, num_workers=0, drop_last=False
        )
        self.valid_dataset = SpeechCommand(valid_dir, self.ver, transform=transform,
                                          embeddings_path=valid_embeddings_path)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=1, num_workers=0)
        self.test_dataset = SpeechCommand(test_dir, self.ver, transform=transform,
                                         embeddings_path=test_embeddings_path)
        self.test_loader = DataLoader(self.test_dataset, batch_size=1, num_workers=0)
        self.plot_loader = DataLoader(self.test_dataset, batch_size=100, num_workers=0)

        print(
            "check num of data train/valid/test %d/%d/%d"
            % (len(self.train_dataset), len(self.valid_dataset), len(self.test_dataset))
        )

        specaugment = self.tau >= 1.5
        frequency_masking_para = {1: 0, 1.5: 1, 2: 3, 3: 5, 6: 7, 8: 7}

        # Define preprocessors
        self.preprocess_train = Preprocess(
            noise_dir,
            self.device,
            specaug=specaugment,
            frequency_masking_para=frequency_masking_para[self.tau],
        )
        self.preprocess_test = Preprocess(noise_dir, self.device)

    def _load_ckpt(self, ckpt_path, model):
        print(f'Loading model: {ckpt_path}')
        ckpt = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        return model

    def _load_model(self):
        """
        Private method that loads the model into the object.
        """

        print("model: BC-ResNet-%.1f+SR on data v0.0%d" % (self.tau, self.ver))
        self.model = BCResNets(int(self.tau * 8)).to(self.device)

        if self.eval or self.plot or self.demo:
            self.model = self._load_ckpt(self.ckpt, self.model)

    def _save_top_3_checkpoints(self, epoch, valid_acc):
        """
        Save checkpoints for top 3 validation accuracies.
        
        Parameters:
            epoch (int): Current training epoch
            valid_acc (float): Validation accuracy for the current epoch
        """

        # Prepare checkpoint dictionary
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'valid_acc': valid_acc
        }

        # If less than 3 best accuracies, always save
        if len(self.top_3_valid_accs) < 3:
            checkpoint_path = os.path.join(self.checkpoint_dir, f'model_epoch_{epoch+1}_acc_{valid_acc:.2f}.ckpt')
            torch.save(checkpoint, checkpoint_path)
            self.top_3_valid_accs.append((valid_acc, checkpoint_path))
            self.top_3_valid_accs.sort(reverse=True)  # Sort in descending order
        else:
            # Check if current accuracy is better than the worst in top 3
            if valid_acc > self.top_3_valid_accs[-1][0]:
                # Remove the worst checkpoint
                _, worst_path = self.top_3_valid_accs.pop()
                os.remove(worst_path)

                # Save new checkpoint
                checkpoint_path = os.path.join(self.checkpoint_dir, f'model_epoch_{epoch+1}_acc_{valid_acc:.2f}.ckpt')
                torch.save(checkpoint, checkpoint_path)
                self.top_3_valid_accs.append((valid_acc, checkpoint_path))
                self.top_3_valid_accs.sort(reverse=True)  # Sort in descending order

        # Log the current top 3 checkpoint paths
        print("Current top 3 validation accuracy checkpoints:")
        for acc, path in self.top_3_valid_accs:
            print(f"Acc: {acc:.3f}, Path: {path}")
    
    def _calculate_params(self, model):
        # Calculate number of parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return total_params, trainable_params

    def _calculate_macs(self, model):
        # Calculate MACs (Multiply-Accumulate Operations)
        input_sample = torch.randn(1, 1, 40, 87).to(self.device)
        input_embedding = torch.randn(1,1,512).to(self.device)
        macs, _ = profile(model, inputs=(input_sample, input_embedding), verbose=False)

        return macs

    def _test_best_checkpoint(self):
        """
        Load and test the best checkpoint from the top 3 validation accuracies.
        """
        if not self.top_3_valid_accs:
            print("No checkpoints were saved. Skipping best checkpoint test.")
            return

        # Sort checkpoints by validation accuracy in descending order
        sorted_checkpoints = sorted(self.top_3_valid_accs, reverse=True)
        
        # Select the best checkpoint
        best_valid_acc, best_checkpoint_path = sorted_checkpoints[0]
        
        print(f"\nTesting best checkpoint with validation accuracy: {best_valid_acc:.3f}")
        print(f"Checkpoint path: {best_checkpoint_path}")

        # Load the best checkpoint
        checkpoint = torch.load(best_checkpoint_path, weights_only=False)
        
        # Create a new model instance and load the state dict
        best_model = BCResNets(int(self.tau * 8)).to(self.device)
        best_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set the model to evaluation mode
        best_model.eval()

        # Replace the current model with the best model for testing
        original_model = self.model
        self.model = best_model

        # Run test on the loaded model
        with torch.no_grad():
            best_test_acc, best_test_auroc, best_test_f1, best_test_fa, best_test_eer = self.Test(self.test_dataset, self.test_loader, augment=False)
            print(f"Best ckpt test - Acc: {best_test_acc:.3f}, AUROC: {best_test_auroc:.3f}, F1: {best_test_f1:.3f}, FA: {best_test_fa:.3f}")
        
        # Calculate number of parameters
        total_params, trainable_params = self._calculate_params(self.model)

        # Calculate MACs (Multiply-Accumulate Operations)
        macs = self._calculate_macs(self.model)

        # Prepare results dictionary
        results = {
            'accuracy': best_test_acc,
            'auroc': best_test_auroc,
            'f1-score': best_test_f1,
            'false_alarm': best_test_fa,
            'EER': best_test_eer,
            'params': {
                'total_params_k': total_params/1000,
                'trainable_params_k': trainable_params/1000
            },
            'macs_m': macs/1e6
        }

        # Save results to JSON
        results_path = os.path.join(self.checkpoint_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)

        # Restore the original model
        self.model = original_model
    def wpl_loss(self, output, target, weights=torch.tensor([1.0, 0.5, 1.0])):
        """Compute the WPL for a sequence.

        Args:
            output (torch.tensor): A tensor containing the model predictions.
            target (torch.tensor): A 1D tensor containing the indices of the target classes.

        Returns:
            torch.tensor: A tensor containing the WPL value for the processed sequence.
        """

        output = torch.exp(output)
        label_mask = F.one_hot(target) > 0.5 # boolean mask
        label_mask_r1 = torch.roll(label_mask, 1, 1) # if ntss, then tss
        label_mask_r2 = torch.roll(label_mask, 2, 1) # if ntss, then ns
        weights = weights.to(self.device)

        # get the probability of the actual label and the other two into one array
        actual = torch.masked_select(output, label_mask)
        plus_one = torch.masked_select(output, label_mask_r1)
        minus_one = torch.masked_select(output, label_mask_r2)

        # arrays of the first pair weight and the second pair weight used in the equation
        w1 = torch.masked_select(weights, label_mask) # if ntss, w1 is <ntss, ns>
        w2 = torch.masked_select(weights, label_mask_r1) # if ntss, w2 is <tss, ntss>

        # first pair
        first_pair = w1 * torch.log(actual / (actual + minus_one))
        second_pair = w2 * torch.log(actual / (actual + plus_one))

        # get the negative mean value for the two pairs
        wpl = -0.5 * (first_pair + second_pair)

        # sum and average for minibatch
        return torch.mean(wpl) 
    
    def Evaluation(self):
        # Calculate number of parameters
        total_params, trainable_params = self._calculate_params(self.model)

        # Calculate MACs (Multiply-Accumulate Operations)
        macs = self._calculate_macs(self.model)

        # Perform evaluation
        with torch.no_grad():
            eval_acc, eval_auroc, eval_f1, eval_fa, eval_eer = self.Test(self.test_dataset, self.test_loader, augment=False)
            
        # Print results
        print(f"Eval - Acc: {eval_acc:.3f}, AUROC: {eval_auroc:.3f}, F1: {eval_f1:.3f}, FA: {eval_fa:.3f}, EER: {eval_eer:.3f}")
        print(f"Params - Total: {total_params/1000:.2f}k, Trainable: {trainable_params/1000:.2f}k")
        print(f"MACs: {macs/1e6:.2f}M")

        # Prepare results dictionary
        results = {
            'accuracy': eval_acc,
            'auroc': eval_auroc,
            'f1-score': eval_f1,
            'false_alarm': eval_fa,
            'EER': eval_eer,
            'params': {
                'total_params_k': total_params/1000,
                'trainable_params_k': trainable_params/1000
            },
            'macs_m': macs/1e6
        }

        # Save results to JSON in the same directory as the checkpoint
        results_path = os.path.join(os.path.dirname(self.ckpt), 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
    
    def _umap(self, embeddings, labels, title, num_classes=None, class_names=None):
        """Generates a 2D UMAP plot for the given embeddings and labels."""

        reducer = umap.UMAP(n_neighbors=5, min_dist=0.5, n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)

        plt.figure(figsize=(8, 6))
        if num_classes is None:
            # Binary classification (Speech vs. Non-Speech, Keyword vs. Non-Keyword)            
            # Add legend based on the title (classifier1 or classifier2)
            if title == "classifier1":
                legend_labels = ["Non-speech", "Speech"]
                cmap = 'cividis'
            elif title == "classifier2":
                legend_labels = ["Non-keyword", "Keyword"]
                cmap = 'RdYlGn'
            else:
                legend_labels = ["0", "1"]  # Default case
                cmap = 'coolwarm'
            scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap=cmap)
            plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels, prop=font_prop)
        else:
            # Multi-class classification (10-class Keyword Classification)
            scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='tab10')
            if class_names:
                plt.legend(handles=scatter.legend_elements()[0], labels=class_names, prop=font_prop)

        plt.xticks([])
        plt.yticks([])
        plt.savefig(f'{self.checkpoint_dir}/{title}.pdf', dpi=800)
        print(f'{self.checkpoint_dir}/{title}.pdf is done!')
        plt.show()
    
    def Plot(self):
        """Loads the model, extracts embeddings, and generates UMAP visualizations."""

        all_labels = []
        all_embeddings = []

        with torch.no_grad():
            for sample in self.plot_loader:
                inputs, speaker_embeddings, labels, speaker_labels = sample
                inputs = inputs.to(self.device)
                inputs = self.preprocess_test(inputs, labels=labels, is_train=False, augment=False)
                embeddings = self.model.encode(inputs)
                all_labels.append(labels.numpy())
                all_embeddings.append(embeddings.cpu().numpy())
        
        
        all_labels = np.concatenate(all_labels, axis=0)
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_embeddings = all_embeddings.reshape(all_embeddings.shape[0], -1)

        # Speech vs. Non-Speech
        speech_labels = (all_labels != 0).astype(int)  # 0 -> non-speech, 1 -> speech
        self._umap(all_embeddings, speech_labels, "classifier1")

        # Keyword vs. Non-Keyword (only process speech samples)
        if (all_labels >= 1).sum() > 0:
            speech_embeddings = all_embeddings[all_labels >= 1]
            keyword_labels = (all_labels[all_labels >= 1] >= 2).astype(int)  # 1 -> non-keyword, 2~11 -> keyword
            self._umap(speech_embeddings, keyword_labels, "classifier2")

        # 10-class Keyword Classification (only process keyword samples)
        if (all_labels >= 2).sum() > 0:
            keyword_embeddings = all_embeddings[all_labels >= 2]
            keyword_class_labels = all_labels[all_labels >= 2] - 2  # Normalize to 0-9
            # class_names = [f"KW {i+1}" for i in range(10)]  # Label keywords as KW_1, KW_2, etc.
            class_names = ['Down', 'Go', 'Left', 'No', 'Off', 'On', 'Right', 'Stop', 'Up', 'Yes']
            self._umap(keyword_embeddings, keyword_class_labels, "classifier3", num_classes=10, class_names=class_names)
    
    def _predict(self, audio_record, audio_upload, threshold):
        # Initialization
        speech_prediction = keyword_prediction = 0.0
        keyword_class_prediction = None
        class_names = {0: 'Down', 1: 'Go', 2: 'Left', 3: 'No', 4: 'Off', 5: 'On', 6: 'Right', 7: 'Stop', 8: 'Up', 9: 'Yes'}

        # Process samples
        audio_input = audio_record if audio_record else audio_upload
        transform = transforms.Compose([Padding()])
        sample, _ = torchaudio.load(audio_input)
        sample = transform(sample)
        sample = self.preprocess_test(x=sample, is_train=False, augment=False)

        with torch.no_grad():
            # Use zero embedding for demo (no speaker verification)
            speaker_embedding = torch.zeros(1, 512).to(self.device)
            probability = self.model.inference(sample, speaker_embedding)
            speech_prediction = probability[0]
            keyword_prediction = probability[1]
            keyword_class_prediction = class_names[torch.argmax(probability[2:]).item()]
        
        yield speech_prediction * 100, keyword_prediction * 100, keyword_class_prediction

    def Demo(self):
        with gr.Blocks() as demo:
            # Title and Description
            gr.Markdown("<h1 style='text-align: center; color: black;'>Keyword Spotting using BC-ResNet with Successive Refinement</h1>")
            gr.Markdown("<h3 style='text-align: center; color: black;'>Record or upload audio to predict if the audio contains keywords.</h3>")
            
            # Interface Layout
            with gr.Row():
                with gr.Column():
                    # Separate recording and file upload
                    record_input = gr.Microphone(type="filepath", label="Record Audio")
                    upload_input = gr.Audio(type="filepath", label="Upload Audio")
                    threshold_input = gr.Slider(minimum=0, maximum=1, value=0.5, step=0.1, label="Threshold")
                with gr.Column():
                    speech_prediction_output = gr.Textbox(label="Speech Prediction (%)")
                    keyword_prediction_output = gr.Textbox(label="Keyword Prediction (%)")
                    keyword_class_prediction_output = gr.Textbox(label="Keyword Class Prediction")
                    
                
            # Prediction Trigger
            predict_btn = gr.Button("Start Prediction")
            predict_btn.click(
                _predict, 
                [record_input, upload_input, threshold_input], 
                [speech_prediction_output, keyword_prediction_output, keyword_class_prediction_output],
                api_name="predict"
            )

        demo.queue()  # Enable queue to support generators
        demo.launch(share=True)



if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    _trainer = Trainer()
    if _trainer.eval:
        _trainer.Evaluation()
    elif _trainer.plot:
        _trainer.Plot()
    elif _trainer.demo:
        _trainer.Demo()
    else:
        _trainer()
