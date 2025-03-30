
import math
import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from typing import List, Optional


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(f"{name}, 'no ignore status'")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_vision_tower_state_maybe_zero_3(named_params, keys_to_match=['']):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class UniVATrainer(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            lr_mapper = {}
            if self.args.mm_projector_lr is not None:
                lr_mapper["mm_projector"] = self.args.mm_projector_lr
            if self.args.mm_vision_tower_lr is not None:
                lr_mapper["vision_tower"] = self.args.mm_vision_tower_lr
            if self.args.mm_inv_projector_lr is not None:
                lr_mapper["mm_inv_projector"] = self.args.mm_inv_projector_lr
            if self.args.mm_eye_projector_lr is not None:
                lr_mapper["mm_eye_projector"] = self.args.mm_eye_projector_lr
            if self.args.mm_mask_projector_lr is not None:
                lr_mapper["mm_mask_projector"] = self.args.mm_mask_projector_lr
            if len(lr_mapper) > 0:
                special_lr_parameters = [name for name, _ in opt_model.named_parameters() if any(module_keyword in name for module_keyword in lr_mapper)]
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]
                if self.args.local_rank == 0 or self.args.local_rank == -1:
                    nospecial_lr_decay = [n for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)]
                    for n in nospecial_lr_decay:
                        print(f'{n:<110}, decay: {self.args.weight_decay}, default lr')
                    nospecial_lr_nodecay = [n for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)]
                    for n in nospecial_lr_nodecay:
                        print(f'{n:<110}, decay: 0.0, default lr')

                for module_keyword, lr in lr_mapper.items():
                    module_parameters = [name for name, _ in opt_model.named_parameters() if module_keyword in name]
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in module_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in module_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": lr,
                            },
                        ]
                    )
                    if self.args.local_rank == 0 or self.args.local_rank == -1:
                        special_lr_decay = [n for n, p in opt_model.named_parameters() if (n in decay_parameters and n in module_parameters and p.requires_grad)]
                        for n in special_lr_decay:
                            print(f'{n:<110}, decay: {self.args.weight_decay}, lr: {lr}')
                        special_lr_nodecay = [n for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in module_parameters and p.requires_grad)]
                        for n in special_lr_nodecay:
                            print(f'{n:<110}, decay: 0.0, lr: {lr}')
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if self.args.local_rank == 0 or self.args.local_rank == -1:
                print(f'{self.optimizer}')
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        if self.args.local_rank == 0 or self.args.local_rank == -1:
                            print(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                if self.args.local_rank == 0 or self.args.local_rank == -1:
                    print(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial):
        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        if self.args.unfreeze_mm_vision_tower and self.args.mm_vision_tower_lr > 0.0:
            model.get_vision_tower().image_processor.save_pretrained(os.path.join(output_dir, 'vision_tower'))
            model.get_vision_tower().vision_tower.config.save_pretrained(os.path.join(output_dir, 'vision_tower'))
            weight_to_save = get_vision_tower_state_maybe_zero_3(model.get_vision_tower().vision_tower.named_parameters())
            weight_to_save = {k.replace('model.vision_tower.vision_tower.', ''): v for k, v in weight_to_save.items()}
            torch.save(weight_to_save, os.path.join(output_dir, 'vision_tower/pytorch_model.bin'))

            self.model.config.mm_vision_tower = os.path.join(output_dir, 'vision_tower')

        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector']
            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)
            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))

            # Only save Inv adapter
            keys_to_match = ['mm_inv_projector']
            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)
            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_inv_projector.bin'))

            # Only save eye adapter
            keys_to_match = ['mm_eye_projector']
            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)
            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_eye_projector.bin'))

            # Only save mask adapter
            keys_to_match = ['mm_mask_projector']
            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)
            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_mask_projector.bin'))
        else:
            self.model.generation_config.do_sample = True
            super(UniVATrainer, self)._save_checkpoint(model, trial)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if self.args.local_rank == 0 or self.args.local_rank == -1:
            print(f"=> saving to {output_dir}")

        
        if self.args.unfreeze_mm_vision_tower and self.args.mm_vision_tower_lr > 0.0:
            self.model.get_vision_tower().image_processor.save_pretrained(os.path.join(output_dir, 'vision_tower'))
            self.model.get_vision_tower().vision_tower.config.save_pretrained(os.path.join(output_dir, 'vision_tower'))
            weight_to_save = get_vision_tower_state_maybe_zero_3(self.model.get_vision_tower().vision_tower.named_parameters())
            weight_to_save = {k.replace('model.vision_tower.vision_tower.', ''): v for k, v in weight_to_save.items()}
            torch.save(weight_to_save, os.path.join(output_dir, 'vision_tower/pytorch_model.bin'))

            self.model.config.mm_vision_tower = os.path.join(output_dir, 'vision_tower')

        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            self.model.generation_config.do_sample = True
            super(UniVATrainer, self)._save(output_dir, state_dict)
            
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)
        if self.state.global_step == 0:
            self.lm_loss = []
        if self.state.global_step == 0:
            self.vm_loss = []
        if self.state.global_step == 0:
            self.eye_loss = []
        if self.state.global_step == 0:
            self.mask_loss = []

        # if self.state.global_step % (self.args.logging_steps * self.args.gradient_accumulation_steps) == 0:
        lm_loss = getattr(outputs, 'lm_loss', False)
        vm_loss = getattr(outputs, 'vm_loss', False)
        eye_loss = getattr(outputs, 'eye_loss', False)
        mask_loss = getattr(outputs, 'mask_loss', False)
        if lm_loss:
            self.lm_loss.append(lm_loss)
        if vm_loss:
            self.vm_loss.append(vm_loss)
        if eye_loss:
            self.eye_loss.append(eye_loss)
        if mask_loss:
            self.mask_loss.append(mask_loss)

        return (loss, outputs) if return_outputs else loss
    
    def log(self, logs, *args, **kwargs):
        if getattr(self, 'lm_loss', False) and len(self.lm_loss) > 0:
            logs["lm_loss"] = round(torch.stack(self.lm_loss).detach().sum().item(), 4)
            self.lm_loss = []
        if getattr(self, 'vm_loss', False) and len(self.vm_loss) > 0:
            logs["vm_loss"] = round(torch.stack(self.vm_loss).detach().mean().item(), 4)
            self.vm_loss = []
        if getattr(self, 'eye_loss', False) and len(self.eye_loss) > 0:
            logs["eye_loss"] = round(torch.stack(self.eye_loss).detach().mean().item(), 4)
            self.eye_loss = []
        if getattr(self, 'mask_loss', False) and len(self.mask_loss) > 0:
            logs["mask_loss"] = round(torch.stack(self.mask_loss).detach().mean().item(), 4)
            self.mask_loss = []
        super().log(logs, *args, **kwargs)
