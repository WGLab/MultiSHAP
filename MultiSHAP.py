#!/usr/bin/env python3
"""
MultiSHAP VQA Analysis Script
Analyzes cross-modal interactions in ViLT VQA models using Shapley Interaction Index.
"""

import argparse
import functools
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import ViltForQuestionAnswering, ViltProcessor
from datasets import load_dataset
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from tqdm import tqdm


class ViLTVQAAnalyzer:
    def __init__(self, model_name: str = "dandelin/vilt-b32-finetuned-vqa", device: str = "mps", model_config: dict = None):
        self.device = device
        self.model_name = model_name
        self.model = ViltForQuestionAnswering.from_pretrained(model_name, use_safetensors=True).to(device)
        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
        self.model.config = model_config
        self.model.eval()
        
        # Get patch configuration
        self.vision_config = self.model.config.vision_config if hasattr(self.model.config, 'vision_config') else self.model.config
        self.image_size = getattr(self.vision_config, 'image_size', 384)  # ViLT default
        self.patch_size = getattr(self.vision_config, 'patch_size', 32)   # ViLT default
        self.patches_per_dim = self.image_size // self.patch_size
        self.num_patches = self.patches_per_dim ** 2
        
        print(f"Loaded ViLT model: {model_name}")
        print(f"Image size: {self.image_size}, Patch size: {self.patch_size}")
        print(f"Patches per dimension: {self.patches_per_dim}")
        print(f"Total patches: {self.num_patches}")
        print(f"Number of answer classes: {self.model.config.num_labels}")

    def create_masked_image(self, original_image: Image.Image, patch_mask: np.ndarray, 
                           mask_value: float = 0.5) -> Image.Image:
        img = original_image.resize((self.image_size, self.image_size))
        img_array = np.array(img).astype(np.float32) / 255.0
        masked_img = img_array.copy()
        
        patch_idx = 0
        for row in range(self.patches_per_dim):
            for col in range(self.patches_per_dim):
                if patch_mask[patch_idx] == 0:  
                    start_row = row * self.patch_size
                    end_row = (row + 1) * self.patch_size
                    start_col = col * self.patch_size
                    end_col = (col + 1) * self.patch_size
                    
                    masked_img[start_row:end_row, start_col:end_col] = mask_value
                
                patch_idx += 1
        
        masked_img = (masked_img * 255).astype(np.uint8)
        return Image.fromarray(masked_img)
    
    def create_masked_question(self, original_question: str, token_mask: list) -> str:
        inputs = self.processor.tokenizer(original_question, return_tensors="pt", padding=True, truncation=True)
        tokens = self.processor.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])[1:-1] #delete first and last token
        
        masked_tokens = []
        for i, token in enumerate(tokens):
            if i < len(token_mask) and token_mask[i] == 1:
                masked_tokens.append(token)
            elif i < len(token_mask):
                masked_tokens.append("[MASK]")
            else:
                masked_tokens.append(token)

        masked_question = self.processor.tokenizer.convert_tokens_to_string(masked_tokens)
        return masked_question.replace("[MASK]", "")
    
    def compute_vqa_logits(self, image: Image.Image, question: str, target_answer_id: int) -> float:
        inputs = self.processor(image, question, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0]  # shape: [num_classes]
            target_logit = logits[target_answer_id].item()
            
        return target_logit

    def get_predicted_answer(self, image: Image.Image, question: str) -> str:
        """Returns the predicted answer string for the given image and question."""
        self.model.eval()
        inputs = self.processor(image, question, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0]  # [num_classes]
            pred_id = logits.argmax().item()
            id2label = self.model.config.id2label
            # id2label may be int->str or str->str, so ensure int
            if isinstance(id2label, dict):
                # Try int key, fallback to str key
                if pred_id in id2label:
                    return id2label[pred_id]
                elif str(pred_id) in id2label:
                    return id2label[str(pred_id)]
                else:
                    return str(pred_id)
            else:
                # fallback: just return the id as string
                return str(pred_id)

    def get_answer_id(self, answer_text: str) -> int:
        answer_to_id = self.model.config.label2id
        return answer_to_id[answer_text.lower()]
    
    def compute_iccs_question_patches(self, 
                                    image: Image.Image, 
                                    question: str,
                                    target_answer: str,
                                    n_samples: int = 200,
                                    patch_indices: list = None,
                                    token_indices: list = None,
                                    use_monte_carlo: bool = True,
                                    stratified_sampling: bool = True) -> tuple:
        """
        Compute the ICCS matrix between image patches and question tokens.
        
        Args:
            image: Input image
            question: Input question
            target_answer: Target answer
            n_samples: Number of Monte Carlo samples (ignored if use_monte_carlo=False)
            patch_indices: Subset of patch indices to compute (for speedup)
            token_indices: Subset of token indices to compute (for speedup)
            use_monte_carlo: If True, use Monte Carlo sampling; if False, compute exact
            stratified_sampling: If True, use stratified sampling for Monte Carlo
        """
        print(f"Computing Question-Patch I_CCS for VQA...")
        print(f"Method: {'Monte Carlo' if use_monte_carlo else 'Exact'}")
        if use_monte_carlo:
            print(f"Sampling: {'Stratified' if stratified_sampling else 'Uniform'}")
            print(f"Number of samples: {n_samples}")
        print(f"Question: {question}")
        print(f"Target Answer: {target_answer}")
        
        target_answer_id = self.get_answer_id(target_answer)
        print(f"Target Answer ID: {target_answer_id}")
        
        inputs = self.processor.tokenizer(question, return_tensors="pt", padding=True)
        tokens = self.processor.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])[1:-1] 
        num_tokens = len(tokens)
        
        print(f"Image patches: {self.num_patches}")
        print(f"Question tokens ({num_tokens}): {tokens}")
        
        # Speedup: allow to only compute a subset of patches/tokens
        if patch_indices is None:
            patch_indices = list(range(self.num_patches))
        if token_indices is None:
            token_indices = list(range(num_tokens))

        iccs_matrix = np.zeros((self.num_patches, num_tokens))

        # Speedup: Use functools.lru_cache to cache masked image/question -> logit
        @functools.lru_cache(maxsize=8192)
        def cached_logits(patch_mask_tuple: tuple, token_mask_tuple: tuple) -> float:
            patch_mask = np.array(patch_mask_tuple)
            token_mask = list(token_mask_tuple)
            masked_image = self.create_masked_image(image, patch_mask)
            masked_question = self.create_masked_question(question, token_mask)
            logit_value = self.compute_vqa_logits(masked_image, masked_question, target_answer_id)
            return logit_value
        
        if use_monte_carlo:
            # Monte Carlo estimation
            print(f"Running Monte Carlo I_CCS computation...")
            iccs_matrix = self._compute_iccs_monte_carlo(
                patch_indices, token_indices, n_samples, cached_logits, 
                stratified_sampling, self.num_patches, num_tokens
            )
        else:
            # Exact computation
            print(f"Running exact I_CCS computation...")
            iccs_matrix = self._compute_iccs_exact(
                patch_indices, token_indices, cached_logits, 
                self.num_patches, num_tokens
            )

        print(f"I_CCS computation completed!")
        print(f"I_CCS statistics:")
        print(f"  Mean: {iccs_matrix.mean():.4f}")
        print(f"  Std: {iccs_matrix.std():.4f}")
        print(f"  Range: [{iccs_matrix.min():.4f}, {iccs_matrix.max():.4f}]")
        
        return iccs_matrix, tokens
    
    def _compute_iccs_exact(self, patch_indices, token_indices, cached_logits, 
                           num_patches, num_tokens):
        """Compute exact ICCS using all possible coalitions."""
        iccs_matrix = np.zeros((num_patches, num_tokens))
        
        total_coalitions = 2 ** (len(patch_indices) + len(token_indices) - 2)
        print(f"Total coalitions to evaluate: {total_coalitions}")
        
        if total_coalitions > 10000:
            print("Warning: Exact computation may be very slow. Consider using Monte Carlo.")
        
        # Generate all possible coalitions (excluding the target patch-token pair)
        for patch_i in tqdm(patch_indices, desc="Exact ICCS"):
            for token_j in token_indices:
                interaction_sum = 0.0
                coalition_count = 0
                
                # All possible coalitions excluding patch_i and token_j
                other_patches = [p for p in patch_indices if p != patch_i]
                other_tokens = [t for t in token_indices if t != token_j]
                
                # Generate all possible combinations
                from itertools import product
                for patch_combo in product([0, 1], repeat=len(other_patches)):
                    for token_combo in product([0, 1], repeat=len(other_tokens)):
                        # Build coalition mask
                        patch_mask = np.ones(num_patches)
                        token_mask = np.ones(num_tokens)
                        
                        # Set other patches according to combination
                        for idx, include in enumerate(patch_combo):
                            if not include:
                                patch_mask[other_patches[idx]] = 0
                        
                        # Set other tokens according to combination
                        for idx, include in enumerate(token_combo):
                            if not include:
                                token_mask[other_tokens[idx]] = 0
                        
                        # Compute the four scenarios for this coalition
                        # Base: neither patch_i nor token_j
                        base_patch_mask = patch_mask.copy()
                        base_patch_mask[patch_i] = 0
                        base_token_mask = token_mask.copy()
                        base_token_mask[token_j] = 0
                        logit_base = cached_logits(tuple(base_patch_mask), tuple(base_token_mask))
                        
                        # Only patch_i
                        patch_only_mask = patch_mask.copy()
                        patch_token_mask = token_mask.copy()
                        patch_token_mask[token_j] = 0
                        logit_patch = cached_logits(tuple(patch_only_mask), tuple(patch_token_mask))
                        
                        # Only token_j
                        token_patch_mask = patch_mask.copy()
                        token_patch_mask[patch_i] = 0
                        token_only_mask = token_mask.copy()
                        logit_token = cached_logits(tuple(token_patch_mask), tuple(token_only_mask))
                        
                        # Both patch_i and token_j
                        both_patch_mask = patch_mask.copy()
                        both_token_mask = token_mask.copy()
                        logit_both = cached_logits(tuple(both_patch_mask), tuple(both_token_mask))
                        
                        # Compute interaction for this coalition
                        coalition_size = sum(patch_combo) + sum(token_combo)
                        total_features = len(other_patches) + len(other_tokens)
                        
                        # Shapley weight
                        if total_features > 0:
                            weight = (1.0 / (total_features + 1)) * (1.0 / (total_features + 1))
                        else:
                            weight = 1.0
                        
                        interaction = (logit_both - logit_patch) - (logit_token - logit_base)
                        interaction_sum += weight * interaction
                        coalition_count += 1
                
                if coalition_count > 0:
                    iccs_matrix[patch_i, token_j] = interaction_sum
        
        return iccs_matrix
    
    def _compute_iccs_monte_carlo(self, patch_indices, token_indices, n_samples, 
                                 cached_logits, stratified_sampling, num_patches, num_tokens):
        """Compute ICCS using Monte Carlo sampling."""
        iccs_matrix = np.zeros((num_patches, num_tokens))
        counts_matrix = np.zeros((num_patches, num_tokens))
        
        if stratified_sampling:
            # Stratified sampling by coalition size
            total_features = len(patch_indices) + len(token_indices) - 2  # Exclude target pair
            strata_samples = max(1, n_samples // (total_features + 1))
            print(f"Using stratified sampling with {strata_samples} samples per stratum")
        
        for patch_i in tqdm(patch_indices, desc="Monte Carlo ICCS"):
            for token_j in token_indices:
                interaction_sum = 0.0
                sample_count = 0
                
                if stratified_sampling:
                    # Sample from each stratum (coalition size)
                    other_patches = [p for p in patch_indices if p != patch_i]
                    other_tokens = [t for t in token_indices if t != token_j]
                    total_others = len(other_patches) + len(other_tokens)
                    
                    for coalition_size in range(total_others + 1):
                        for _ in range(strata_samples):
                            # Generate random coalition of specific size
                            coalition = self._generate_random_coalition(
                                other_patches, other_tokens, coalition_size
                            )
                            
                            interaction = self._compute_single_interaction(
                                coalition, patch_i, token_j, cached_logits, 
                                num_patches, num_tokens
                            )
                            
                            interaction_sum += interaction
                            sample_count += 1
                else:
                    # Uniform random sampling
                    other_patches = [p for p in patch_indices if p != patch_i]
                    other_tokens = [t for t in token_indices if t != token_j]
                    
                    for _ in range(n_samples):
                        # Generate completely random coalition
                        coalition_size = np.random.randint(0, len(other_patches) + len(other_tokens) + 1)
                        coalition = self._generate_random_coalition(
                            other_patches, other_tokens, coalition_size
                        )
                        
                        interaction = self._compute_single_interaction(
                            coalition, patch_i, token_j, cached_logits, 
                            num_patches, num_tokens
                        )
                        
                        interaction_sum += interaction
                        sample_count += 1
                
                if sample_count > 0:
                    iccs_matrix[patch_i, token_j] = interaction_sum / sample_count
                    counts_matrix[patch_i, token_j] = sample_count
        
        return iccs_matrix
    
    def _generate_random_coalition(self, patches, tokens, coalition_size):
        """Generate a random coalition of specified size."""
        all_features = patches + tokens
        if coalition_size > len(all_features):
            coalition_size = len(all_features)
        
        coalition_features = np.random.choice(
            all_features, size=coalition_size, replace=False
        ) if len(all_features) > 0 else []
        
        return {
            'patches': [p for p in coalition_features if p in patches],
            'tokens': [t for t in coalition_features if t in tokens]
        }
    
    def _compute_single_interaction(self, coalition, patch_i, token_j, cached_logits, 
                                   num_patches, num_tokens):
        """Compute interaction for a single coalition."""
        # Base coalition mask (excluding target patch and token)
        base_patch_mask = np.ones(num_patches)
        base_token_mask = np.ones(num_tokens)
        
        # Set coalition
        for p in range(num_patches):
            if p not in coalition['patches'] and p != patch_i:
                base_patch_mask[p] = 0
        for t in range(num_tokens):
            if t not in coalition['tokens'] and t != token_j:
                base_token_mask[t] = 0
        
        # Exclude target patch and token for base
        base_patch_mask[patch_i] = 0
        base_token_mask[token_j] = 0
        logit_base = cached_logits(tuple(base_patch_mask), tuple(base_token_mask))
        
        # Include only patch_i
        patch_mask = base_patch_mask.copy()
        patch_mask[patch_i] = 1
        token_mask = base_token_mask.copy()
        logit_patch = cached_logits(tuple(patch_mask), tuple(token_mask))
        
        # Include only token_j
        patch_mask = base_patch_mask.copy()
        token_mask = base_token_mask.copy()
        token_mask[token_j] = 1
        logit_token = cached_logits(tuple(patch_mask), tuple(token_mask))
        
        # Include both patch_i and token_j
        patch_mask = base_patch_mask.copy()
        patch_mask[patch_i] = 1
        token_mask = base_token_mask.copy()
        token_mask[token_j] = 1
        logit_both = cached_logits(tuple(patch_mask), tuple(token_mask))
        
        # Compute interaction
        interaction = (logit_both - logit_patch) - (logit_token - logit_base)
        return interaction

    def visualize_question_token_heatmaps(self, image: Image.Image, 
                                        iccs_matrix: np.ndarray, 
                                        tokens: list, 
                                        question: str,
                                        target_answer: str,
                                        max_tokens: int = 40,
                                        patch_stride: int = 1,
                                        visualize_average_only: bool = False,
                                        save_path: Optional[str] = None):
        """Visualize ICCS heatmaps."""
        n_tokens = min(len(tokens), max_tokens)
        show_average = n_tokens > 1

        if visualize_average_only:
            total_plots = 1
        else:
            total_plots = n_tokens + (1 if show_average else 0)
        ncols = min(total_plots, 4)
        nrows = (total_plots + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
        if nrows == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        plot_idx = 0

        if not visualize_average_only:
            for token_idx in range(n_tokens):
                iccs_vector = iccs_matrix[:, token_idx]
                # Speedup: Optionally, subsample patches for visualization
                if patch_stride > 1:
                    iccs_vector = iccs_vector[::patch_stride]
                    patches_per_dim = self.patches_per_dim // patch_stride
                else:
                    patches_per_dim = self.patches_per_dim
                interaction_grid = iccs_vector.reshape(patches_per_dim, patches_per_dim)
                interaction_grid = np.ascontiguousarray(interaction_grid, dtype=np.float32)
                
                img_resized = image.resize((self.image_size, self.image_size))
                img_array = np.asarray(img_resized, dtype=np.float32) / 255.0

                upsampled = cv2.resize(
                    interaction_grid, 
                    (self.image_size, self.image_size), 
                    interpolation=cv2.INTER_CUBIC
                )
                blur_ksize = int(self.image_size // 8) | 1
                if blur_ksize > 1:
                    upsampled = cv2.GaussianBlur(upsampled, (blur_ksize, blur_ksize), 0)

                abs_max = max(abs(upsampled.min()), abs(upsampled.max()))
                if abs_max > 1e-10:
                    interaction_norm = (upsampled + abs_max) / (2 * abs_max)
                else:
                    interaction_norm = np.ones_like(upsampled) * 0.5
                interaction_norm = np.clip(interaction_norm, 0, 1)
                
                cmap = cm.get_cmap('RdBu_r')
                interaction_colored = cmap(interaction_norm)
                if interaction_colored.shape[-1] == 4:
                    interaction_colored = interaction_colored[:, :, :3]
                interaction_colored = np.ascontiguousarray(interaction_colored, dtype=np.float32)

                alpha_map = np.clip(np.abs(upsampled) / (abs_max + 1e-8), 0.40, 0.90)
                alpha = 0.7 * alpha_map

                if img_array.shape[:2] != interaction_colored.shape[:2]:
                    img_array = np.asarray(
                        Image.fromarray((img_array * 255).astype(np.uint8)).resize(
                            interaction_colored.shape[:2][::-1], resample=Image.BICUBIC
                        )
                    ) / 255.0

                blended = (1 - alpha[..., None]) * img_array + alpha[..., None] * interaction_colored
                blended = np.ascontiguousarray(np.clip(blended, 0, 1), dtype=np.float32)
                
                ax = axes[plot_idx]
                im = ax.imshow(blended)
                
                clean_token = tokens[token_idx].replace('Ġ', '').replace('##', '')
                ax.set_title(f'Token: "{clean_token}"\nRange: [{iccs_vector.min():.3f}, {iccs_vector.max():.3f}]')
                ax.axis('off')
                
                norm = Normalize(vmin=-abs_max, vmax=abs_max)
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Logit Interaction', rotation=270, labelpad=15)
                plot_idx += 1

        # Always plot average if show_average is True (n_tokens > 1)
        if show_average and (plot_idx < len(axes)):
            mean_iccs = iccs_matrix[:, :n_tokens].mean(axis=1)
            if patch_stride > 1:
                mean_iccs = mean_iccs[::patch_stride]
                patches_per_dim = self.patches_per_dim // patch_stride
            else:
                patches_per_dim = self.patches_per_dim
            mean_grid = mean_iccs.reshape(patches_per_dim, patches_per_dim)
            mean_grid = np.ascontiguousarray(mean_grid, dtype=np.float32)
            
            img_resized = image.resize((self.image_size, self.image_size))
            img_array = np.asarray(img_resized, dtype=np.float32) / 255.0
            
            upsampled = cv2.resize(mean_grid, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
            blur_ksize = int(self.image_size // 8) | 1
            if blur_ksize > 1:
                upsampled = cv2.GaussianBlur(upsampled, (blur_ksize, blur_ksize), 0)

            abs_max = max(abs(upsampled.min()), abs(upsampled.max()))
            if abs_max > 1e-10:
                interaction_norm = (upsampled + abs_max) / (2 * abs_max)
            else:
                interaction_norm = np.ones_like(upsampled) * 0.5
            interaction_norm = np.clip(interaction_norm, 0, 1)
            
            cmap = cm.get_cmap('RdBu_r')
            interaction_colored = cmap(interaction_norm)
            if interaction_colored.shape[-1] == 4:
                interaction_colored = interaction_colored[:, :, :3]
            interaction_colored = np.ascontiguousarray(interaction_colored, dtype=np.float32)

            alpha_map = np.clip(np.abs(upsampled) / (abs_max + 1e-8), 0.40, 0.90)
            alpha = 0.7 * alpha_map

            if img_array.shape[:2] != interaction_colored.shape[:2]:
                img_array = np.asarray(
                    Image.fromarray((img_array * 255).astype(np.uint8)).resize(
                        interaction_colored.shape[:2][::-1], resample=Image.BICUBIC
                    )
                ) / 255.0

            blended = (1 - alpha[..., None]) * img_array + alpha[..., None] * interaction_colored
            blended = np.ascontiguousarray(np.clip(blended, 0, 1), dtype=np.float32)
            
            ax = axes[plot_idx]
            im = ax.imshow(blended)
            ax.set_title(f'Average over {n_tokens} tokens\nRange: [{mean_iccs.min():.3f}, {mean_iccs.max():.3f}]')
            ax.axis('off')
            norm = Normalize(vmin=-abs_max, vmax=abs_max)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Logit Interaction', rotation=270, labelpad=15)
            plot_idx += 1

        for i in range(plot_idx, len(axes)):
            axes[i].axis('off')
            
        plt.suptitle(f'Question-Patch Interactions\nQ: "{question}" | A: "{target_answer}"', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to '{save_path}'")
        else:
            try:
                plt.show()
            except Exception as e:
                print(f"Display error: {e}")
                default_path = 'vqa_question_patch_interactions.png'
                plt.savefig(default_path, dpi=150, bbox_inches='tight')
                print(f"Saved visualization to '{default_path}'")
        
        plt.close(fig)


def analyze_vilt_vqa_iccs_dataset(
    dataset, 
    analyzer, 
    num_samples=100, 
    n_iccs_samples=128, 
    visualize_first=3, 
    sample_indices=None, 
    random_seed=42,
    visualize_average_only=False,
    output_dir: Optional[str] = None,
    use_monte_carlo=True,
    stratified_sampling=True
):
    """Perform dataset-level analysis of ViLT VQA I_CCS interactions."""
    print(f"🚀 Running dataset-level ViLT VQA I_CCS analysis on {num_samples} samples...")

    results = []
    if sample_indices is not None:
        indices = list(sample_indices)[:num_samples]
    else:
        indices = list(range(len(dataset)))
        random.Random(random_seed).shuffle(indices)
        indices = indices[:num_samples]

    # For new stats
    all_pos_sums = []
    all_neg_sums = []
    all_pos_ratios = []
    pos_larger_than_neg_count = 0
    total_valid_samples = 0
    correct_count = 0

    for idx, i in enumerate(indices):
        example = dataset[i]
        image = example['image']
        question = example['question']
        target_answer = example['multiple_choice_answer']

        print(f"\n[{idx+1}/{num_samples}] Q: {question} | A: {target_answer}")

        try:
            target_answer_id = analyzer.get_answer_id(target_answer)
            baseline_logit = analyzer.compute_vqa_logits(image, question, target_answer_id)

            iccs_matrix, tokens = analyzer.compute_iccs_question_patches(
                image, question, target_answer, 
                n_samples=n_iccs_samples,
                use_monte_carlo=use_monte_carlo,
                stratified_sampling=stratified_sampling
            )

            # Compute stats
            max_iccs = float(iccs_matrix.max())
            min_iccs = float(iccs_matrix.min())
            mean_abs_iccs = float(np.abs(iccs_matrix).mean())
            token_importance = np.abs(iccs_matrix).sum(axis=0)
            top_tokens = np.argsort(token_importance)[-3:][::-1]
            top_tokens_str = [tokens[token_idx].replace('Ġ', '').replace('##', '') for token_idx in top_tokens]

            # Sum positive and negative elements, compute ratio
            pos_sum = float(np.sum(iccs_matrix[iccs_matrix > 0]))
            neg_sum = float(np.sum(iccs_matrix[iccs_matrix < 0]))
            abs_pos_sum = abs(pos_sum)
            abs_neg_sum = abs(neg_sum)
            denom = abs_pos_sum + abs_neg_sum
            if denom > 0:
                pos_ratio = pos_sum / denom
            else:
                pos_ratio = 0.0  # or np.nan

            all_pos_sums.append(pos_sum)
            all_neg_sums.append(neg_sum)
            all_pos_ratios.append(pos_ratio)
            total_valid_samples += 1
            if abs_pos_sum > abs_neg_sum:
                pos_larger_than_neg_count += 1

            # Compute accuracy: check if predicted answer matches ground truth
            try:
                predicted_answer = analyzer.get_predicted_answer(image, question)
                is_correct = (str(predicted_answer).strip().lower() == str(target_answer).strip().lower())
            except Exception:
                is_correct = False
            if is_correct:
                correct_count += 1

            result = {
                'index': i,
                'question': question,
                'answer': target_answer,
                'baseline_logit': float(baseline_logit),
                'max_iccs': max_iccs,
                'min_iccs': min_iccs,
                'mean_abs_iccs': mean_abs_iccs,
                'top_tokens': top_tokens_str,
                'top_token_importance': [float(token_importance[token_idx]) for token_idx in top_tokens],
                'iccs_pos_sum': pos_sum,
                'iccs_neg_sum': neg_sum,
                'iccs_pos_ratio': pos_ratio,
                'iccs_pos_larger_than_neg': abs_pos_sum > abs_neg_sum,
                'is_correct': is_correct,
            }
            results.append(result)

            if idx < visualize_first:
                print(f"  [Visualization] Creating question-token heatmaps...")
                save_path = None
                if output_dir:
                    output_path = Path(output_dir)
                    output_path.mkdir(exist_ok=True)
                    save_path = output_path / f"sample_{idx+1}_interactions.png"
                
                analyzer.visualize_question_token_heatmaps(
                    image, iccs_matrix, tokens, question, target_answer,
                    visualize_average_only=visualize_average_only,
                    save_path=str(save_path) if save_path else None
                )

            print(f"  [Stats] Max: {max_iccs:.4f} | Min: {min_iccs:.4f} | MeanAbs: {mean_abs_iccs:.4f}")
            print(f"  [Tokens] Top: {top_tokens_str}")
            print(f"  [ICCS] Pos sum: {pos_sum:.4f} | Neg sum: {neg_sum:.4f} | Pos ratio: {pos_ratio:.4f} | Pos>Neg: {abs_pos_sum > abs_neg_sum}")
            print(f"  [Accuracy] Predicted: {predicted_answer if 'predicted_answer' in locals() else 'N/A'} | Correct: {is_correct}")

        except Exception as e:
            print(f"❌ I_CCS computation failed for example {i}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Dataset-level summary
    if results:
        all_mean_abs = [r['mean_abs_iccs'] for r in results]
        all_max = [r['max_iccs'] for r in results]
        all_min = [r['min_iccs'] for r in results]
        print("\n📊 Dataset-level I_CCS summary:")
        print(f"  Mean of mean-abs: {np.mean(all_mean_abs):.4f}")
        print(f"  Mean of max: {np.mean(all_max):.4f}")
        print(f"  Mean of min: {np.mean(all_min):.4f}")

        # ICCS positive/negative stats
        print("\n📈 ICCS Positive/Negative stats across dataset:")
        print(f"  Mean positive sum: {np.mean(all_pos_sums):.4f}")
        print(f"  Mean negative sum: {np.mean(all_neg_sums):.4f}")
        print(f"  Mean positive ratio: {np.mean(all_pos_ratios):.4f}")
        if total_valid_samples > 0:
            prop_pos_larger = pos_larger_than_neg_count / total_valid_samples
            print(f"  Proportion of samples with |pos_sum| > |neg_sum|: {prop_pos_larger:.4f}")
            accuracy = correct_count / total_valid_samples
            print(f"  Accuracy: {accuracy:.4f}")
        else:
            print("  No valid samples for positive/negative ICCS stats.")
    else:
        print("No successful results to summarize.")

    return results, indices


def main():
    parser = argparse.ArgumentParser(
        description="MultiSHAP VQA Analysis: Analyze cross-modal interactions in ViLT VQA models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument(
        '--model-name', 
        type=str, 
        default='dandelin/vilt-b32-finetuned-vqa',
        help='Hugging Face model name for ViLT VQA model'
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default='auto',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        help='Device to run the model on. "auto" will select the best available device.'
    )
    
    # Dataset arguments
    parser.add_argument(
        '--dataset-name', 
        type=str, 
        default='HuggingFaceM4/VQAv2',
        help='Hugging Face dataset name'
    )
    parser.add_argument(
        '--dataset-split', 
        type=str, 
        default='validation',
        help='Dataset split to use'
    )
    parser.add_argument(
        '--num-samples', 
        type=int, 
        default=10,
        help='Number of samples to analyze from the dataset'
    )
    parser.add_argument(
        '--sample-indices', 
        type=int, 
        nargs='*',
        help='Specific indices to analyze (if not provided, will sample randomly)'
    )
    parser.add_argument(
        '--random-seed', 
        type=int, 
        default=42,
        help='Random seed for reproducible sampling'
    )
    
    # Analysis arguments
    parser.add_argument(
        '--n-iccs-samples', 
        type=int, 
        default=128,
        help='Number of Monte Carlo samples for ICCS computation (ignored if --exact is used)'
    )
    parser.add_argument(
        '--exact', 
        action='store_true',
        help='Use exact Shapley computation instead of Monte Carlo (much slower but precise)'
    )
    parser.add_argument(
        '--no-stratified', 
        action='store_true',
        help='Use uniform sampling instead of stratified sampling for Monte Carlo'
    )
    parser.add_argument(
        '--visualize-first', 
        type=int, 
        default=3,
        help='Number of examples to visualize'
    )
    parser.add_argument(
        '--visualize-average-only', 
        action='store_true',
        help='Only visualize average token interactions (faster)'
    )
    parser.add_argument(
        '--max-tokens', 
        type=int, 
        default=20,
        help='Maximum number of tokens to visualize individually'
    )
    
    # Output arguments
    parser.add_argument(
        '--output-dir', 
        type=str,
        help='Directory to save visualization outputs (if not provided, will try to display)'
    )
    parser.add_argument(
        '--save-results', 
        type=str,
        help='Path to save analysis results as JSON'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        print(f"Auto-detected device: {device}")
    else:
        device = args.device
    
    # Initialize analyzer
    print(f"Initializing ViLT VQA Analyzer with model: {args.model_name}")
    analyzer = ViLTVQAAnalyzer(model_name=args.model_name, device=device)
    
    # Load dataset
    print(f"Loading dataset: {args.dataset_name} ({args.dataset_split} split)")
    try:
        dataset = load_dataset(args.dataset_name, split=args.dataset_split)
        print(f"Dataset loaded successfully. Total samples: {len(dataset)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    # Validate sample indices
    if args.sample_indices:
        invalid_indices = [idx for idx in args.sample_indices if idx >= len(dataset)]
        if invalid_indices:
            print(f"Warning: Invalid indices {invalid_indices} (dataset has {len(dataset)} samples)")
            args.sample_indices = [idx for idx in args.sample_indices if idx < len(dataset)]
        if not args.sample_indices:
            print("No valid indices provided. Using random sampling.")
            args.sample_indices = None
    
    # Create output directory if specified
    if args.output_dir:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_path}")
    
    # Run analysis
    print(f"\n{'='*60}")
    print("Starting MultiSHAP VQA Analysis")
    print(f"{'='*60}")
    
    try:
        results, indices = analyze_vilt_vqa_iccs_dataset(
            dataset=dataset,
            analyzer=analyzer,
            num_samples=args.num_samples,
            n_iccs_samples=args.n_iccs_samples,
            visualize_first=args.visualize_first,
            sample_indices=args.sample_indices,
            random_seed=args.random_seed,
            visualize_average_only=args.visualize_average_only,
            output_dir=args.output_dir,
            use_monte_carlo=not args.exact,
            stratified_sampling=not args.no_stratified
        )
        
        print(f"\n{'='*60}")
        print("Analysis completed successfully!")
        print(f"{'='*60}")
        print(f"Processed {len(results)} samples")
        print(f"Indices used: {indices[:10]}{'...' if len(indices) > 10 else ''}")
        
        # Save results if requested
        if args.save_results:
            import json
            save_path = Path(args.save_results)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert numpy types to native Python types for JSON serialization
            json_results = []
            for result in results:
                json_result = {}
                for key, value in result.items():
                    if isinstance(value, np.ndarray):
                        json_result[key] = value.tolist()
                    elif isinstance(value, (np.integer, np.floating)):
                        json_result[key] = value.item()
                    else:
                        json_result[key] = value
                json_results.append(json_result)
            
            analysis_data = {
                'args': vars(args),
                'results': json_results,
                'indices': indices,
                'summary': {
                    'total_samples': len(results),
                    'mean_pos_ratio': np.mean([r['iccs_pos_ratio'] for r in results]) if results else 0,
                    'accuracy': len([r for r in results if r['is_correct']]) / len(results) if results else 0
                }
            }
            
            with open(save_path, 'w') as f:
                json.dump(analysis_data, f, indent=2)
            print(f"Results saved to: {save_path}")
        
        # Print summary statistics
        if results:
            print(f"\nSummary Statistics:")
            print(f"  Average ICCS positive ratio: {np.mean([r['iccs_pos_ratio'] for r in results]):.4f}")
            print(f"  Accuracy: {len([r for r in results if r['is_correct']]) / len(results):.4f}")
            print(f"  Average mean absolute ICCS: {np.mean([r['mean_abs_iccs'] for r in results]):.4f}")
            
            # Top tokens analysis
            all_top_tokens = []
            for result in results:
                all_top_tokens.extend(result['top_tokens'])
            
            if all_top_tokens:
                from collections import Counter
                token_counts = Counter(all_top_tokens)
                print(f"\nMost frequent important tokens:")
                for token, count in token_counts.most_common(10):
                    print(f"    {token}: {count} times")
    
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during analysis: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()