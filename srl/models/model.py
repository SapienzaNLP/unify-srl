from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AdamW

from layers.word_encoder import WordEncoder
from layers.sequence_encoder import SequenceEncoder
from layers.state_encoder import StateEncoder


class SimpleModel(pl.LightningModule):

    def __init__(self, hparams, padding_token_id=0, padding_target_id=-1):
        super(SimpleModel, self).__init__()
        self.hparams = hparams
        self.padding_token_id = padding_token_id
        self.padding_target_id = padding_target_id
        self.num_roles = self.hparams.num_roles
        self.num_senses = self.hparams.num_senses

        self.word_encoder = WordEncoder(self.hparams, padding_target_id)
        word_embedding_size = self.word_encoder.word_embedding_size

        self.sequence_encoder = SequenceEncoder(
            encoder_type=self.hparams.sequence_representation,
            lstm_input_size=word_embedding_size,
            lstm_hidden_size=self.hparams.lstm_hidden_size,
            lstm_num_layers=self.hparams.lstm_num_layers,
            lstm_dropout=self.hparams.lstm_dropout,
            lstm_bidirectional=self.hparams.lstm_bidirectional,
            pack_sequences=True)
        sequence_state_size = self.sequence_encoder.sequence_state_size

        self.predicate_encoder = StateEncoder(
            sequence_state_size,
            self.hparams.predicate_encoding_size,
            self.hparams.predicate_encoder_layers,
            self.hparams.predicate_encoder_activation,
            self.hparams.predicate_dropout_rate)

        self.sense_encoder = StateEncoder(
            sequence_state_size,
            self.hparams.sense_encoding_size,
            self.hparams.sense_encoder_layers,
            self.hparams.sense_encoder_activation,
            self.hparams.sense_dropout_rate)

        self.predicate_timestep_encoder = StateEncoder(
            sequence_state_size,
            self.hparams.predicate_timestep_encoding_size,
            self.hparams.predicate_timestep_encoder_layers,
            self.hparams.predicate_timestep_encoder_activation,
            self.hparams.predicate_timestep_dropout_rate)

        self.argument_timestep_encoder = StateEncoder(
            sequence_state_size,
            self.hparams.argument_timestep_encoding_size,
            self.hparams.argument_timestep_encoder_layers,
            self.hparams.argument_timestep_encoder_activation,
            self.hparams.argument_timestep_dropout_rate)

        self.argument_encoder = StateEncoder(
            self.hparams.predicate_timestep_encoding_size + self.hparams.argument_timestep_encoding_size,
            self.hparams.argument_encoding_size,
            self.hparams.argument_encoder_layers,
            self.hparams.argument_encoder_activation,
            self.hparams.argument_dropout_rate)

        self.argument_sequence_encoder = SequenceEncoder(
            encoder_type=self.hparams.sequence_representation,
            lstm_input_size=self.hparams.argument_encoding_size,
            lstm_hidden_size=self.hparams.argument_lstm_hidden_size,
            lstm_num_layers=self.hparams.argument_lstm_num_layers,
            lstm_dropout=self.hparams.argument_lstm_dropout,
            lstm_bidirectional=self.hparams.argument_lstm_bidirectional,
            pack_sequences=True)
        argument_sequence_state_size = self.argument_sequence_encoder.sequence_state_size

        self.predicate_scorer = nn.Linear(self.hparams.predicate_encoding_size, 2)
        self.sense_scorer = nn.Linear(self.hparams.sense_encoding_size, self.num_senses)
        self.role_scorer = nn.Linear(argument_sequence_state_size, self.num_roles)

        self.role_weights = torch.as_tensor(self.hparams.role_weights).to('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        word_ids = x['word_ids']
        subword_indices = x['subword_indices']
        sequence_lengths = x['sequence_lengths']
        tokenized_sequence_lengths = x['tokenized_sequence_lengths']
        predicate_indices = x['predicate_indices']

        word_embeddings = self.word_encoder(word_ids, subword_indices=subword_indices, sequence_lengths=tokenized_sequence_lengths)

        sequence_states = self.sequence_encoder(word_embeddings, sequence_lengths)

        predicate_encodings = self.predicate_encoder(sequence_states)
        predicate_scores = self.predicate_scorer(predicate_encodings)
        sense_encodings = self.sense_encoder(sequence_states)
        sense_encodings = sense_encodings[predicate_indices]
        sense_scores = self.sense_scorer(sense_encodings)

        timesteps = sequence_states.shape[1]

        predicate_timestep_encodings = self.predicate_timestep_encoder(sequence_states)
        predicate_timestep_encodings = predicate_timestep_encodings.unsqueeze(2).expand(-1, -1, timesteps, -1)

        argument_timestep_encodings = self.argument_timestep_encoder(sequence_states)
        argument_timestep_encodings = argument_timestep_encodings.unsqueeze(1).expand(-1, timesteps, -1, -1)

        predicate_argument_states = torch.cat((predicate_timestep_encodings, argument_timestep_encodings), dim=-1)
        predicate_argument_states = predicate_argument_states[predicate_indices]

        argument_sequence_lengths = sequence_lengths[predicate_indices[0]]
        max_argument_sequence_length = torch.max(argument_sequence_lengths)
        predicate_argument_states = predicate_argument_states[:, :max_argument_sequence_length, :]
        argument_encodings = self.argument_encoder(predicate_argument_states)
        argument_encodings = self.argument_sequence_encoder(argument_encodings, argument_sequence_lengths)
        role_scores = self.role_scorer(argument_encodings)

        return {
            'predicates': predicate_scores,
            'senses': sense_scores,
            'roles': role_scores,
        }

    def configure_optimizers(self):
        base_parameters = []
        base_parameters.extend(list(self.sequence_encoder.parameters()))
        base_parameters.extend(list(self.predicate_encoder.parameters()))
        base_parameters.extend(list(self.sense_encoder.parameters()))
        base_parameters.extend(list(self.predicate_timestep_encoder.parameters()))
        base_parameters.extend(list(self.argument_timestep_encoder.parameters()))
        base_parameters.extend(list(self.argument_encoder.parameters()))
        base_parameters.extend(list(self.argument_sequence_encoder.parameters()))
        base_parameters.extend(list(self.predicate_scorer.parameters()))
        base_parameters.extend(list(self.sense_scorer.parameters()))
        base_parameters.extend(list(self.role_scorer.parameters()))

        language_model_parameters = []
        for parameter_name, parameter in self.word_encoder.named_parameters():
            if 'word_embedding' not in parameter_name:
                base_parameters.append(parameter)
            else:
                language_model_parameters.append(parameter)

        optimizer = torch.optim.AdamW(
            [
                {'params': base_parameters},
                {'params': language_model_parameters, 'lr': self.hparams.language_model_learning_rate, 'weight_decay': self.hparams.language_model_weight_decay, 'correct_bias': False},
            ],
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None, using_native_amp=None):
        step = self.trainer.global_step
        warmup_steps = self.hparams.warmup_epochs * self.hparams.steps_per_epoch
        cooldown_steps = warmup_steps + self.hparams.cooldown_epochs * self.hparams.steps_per_epoch
        training_steps = self.hparams.max_epochs * self.hparams.steps_per_epoch

        if step < warmup_steps:
            lr_scale = min(1., float(step + 1) / warmup_steps)
            optimizer.param_groups[0]['lr'] = lr_scale * self.hparams.learning_rate
            optimizer.param_groups[1]['lr'] = lr_scale * self.hparams.language_model_learning_rate

        elif step < cooldown_steps:
            progress = float(step - warmup_steps) / float(max(1, cooldown_steps - warmup_steps))
            lr_scale = (1. - progress)
            optimizer.param_groups[0]['lr'] = self.hparams.min_learning_rate + lr_scale * (self.hparams.learning_rate - self.hparams.min_learning_rate)
            optimizer.param_groups[1]['lr'] = self.hparams.language_model_min_learning_rate + lr_scale * (self.hparams.language_model_learning_rate - self.hparams.language_model_min_learning_rate)

        else:
            progress = float(step - cooldown_steps) / float(max(1, training_steps - cooldown_steps))
            lr_scale = (1. - progress)
            optimizer.param_groups[0]['lr'] = lr_scale * self.hparams.min_learning_rate
            optimizer.param_groups[1]['lr'] = lr_scale * self.hparams.language_model_min_learning_rate

        # Update params.
        optimizer.step()
        optimizer.zero_grad()

    def training_step(self, batch, batch_index):
        sample, labels = batch
        scores = self(sample)

        labels['senses'] = labels['senses'][sample['predicate_indices']]
        sample['word_sense_mask'] = sample['word_sense_mask'][sample['predicate_indices']]

        labels['roles'] = labels['roles'][sample['predicate_indices']]
        argument_sequence_lengths = sample['sequence_lengths'][sample['predicate_indices'][0]]
        max_argument_sequence_length = torch.max(argument_sequence_lengths)
        labels['roles'] = labels['roles'][:, 1:max_argument_sequence_length-1].contiguous()
        scores['roles'] = scores['roles'][:, 1:max_argument_sequence_length-1:, :].contiguous()

        predicate_identification_loss = SimpleModel._compute_classification_loss(
            scores['predicates'],
            labels['predicates'],
            2,
            ignore_index=self.padding_target_id,
        )
        sense_classification_loss = SimpleModel._compute_classification_loss(
            scores['senses'],
            labels['senses'],
            self.num_senses,
            ignore_index=self.padding_target_id,
        )

        if self.hparams.argument_classifier == 'softmax':
            argument_classification_loss = SimpleModel._compute_classification_loss(
                scores['roles'],
                labels['roles'],
                self.num_roles,
                ignore_index=self.padding_target_id,
                weight=self.role_weights,
            )

        loss = predicate_identification_loss \
            + sense_classification_loss \
            + argument_classification_loss

        if torch.isnan(loss):
            print('Loss:', loss)
            print('Predicate identification loss:', predicate_identification_loss)
            print('Predicate disambiguation loss:', sense_classification_loss)
            print('Argument classification loss:', argument_classification_loss)
            raise ValueError('NaN loss during training!')

        tensorboard_logs = {
            'train_loss': loss,
            'train_loss_predicate_identification': predicate_identification_loss,
            'train_loss_sense_classification': sense_classification_loss,
            'train_loss_argument_classification': argument_classification_loss,
        }

        return {
            'loss': loss,
            'log': tensorboard_logs,
        }

    def validation_step(self, batch, batch_index):
        return self._shared_step(batch, batch_index)

    def test_step(self, batch, batch_index):
        return self._shared_step(batch, batch_index)

    def _shared_step(self, batch, batch_index):
        sample, labels = batch
        scores = self(sample)

        labels['senses'] = labels['senses'][sample['predicate_indices']]
        sample['word_sense_mask'] = sample['word_sense_mask'][sample['predicate_indices']]

        labels['roles'] = labels['roles'][sample['predicate_indices']]
        argument_sequence_lengths = sample['sequence_lengths'][sample['predicate_indices'][0]]
        max_argument_sequence_length = torch.max(argument_sequence_lengths)
        labels['roles'] = labels['roles'][:, 1:max_argument_sequence_length-1].contiguous()
        scores['roles'] = scores['roles'][:, 1:max_argument_sequence_length-1:, :].contiguous()

        predicate_identification_loss = SimpleModel._compute_classification_loss(
            scores['predicates'],
            labels['predicates'],
            2,
            ignore_index=self.padding_target_id,
        )
        sense_classification_loss = SimpleModel._compute_classification_loss(
            scores['senses'],
            labels['senses'],
            self.num_senses,
            ignore_index=self.padding_target_id,
        )
        if self.hparams.argument_classifier == 'softmax':
            argument_classification_loss = SimpleModel._compute_classification_loss(
                scores['roles'],
                labels['roles'],
                self.num_roles,
                ignore_index=self.padding_target_id,
                weight=self.role_weights,
            )

        loss = predicate_identification_loss \
            + sense_classification_loss \
            + argument_classification_loss
        metrics = self.compute_step_metrics(scores, labels)

        if torch.isnan(loss) or not torch.isfinite(loss):
            print('Loss:', loss)
            print('Predicate identification loss:', predicate_identification_loss)
            print('Predicate disambiguation loss:', sense_classification_loss)
            print('Argument classification loss:', argument_classification_loss)
            raise ValueError('NaN loss during training!')

        return {
            'loss': loss,
            'predicate_identification_loss': predicate_identification_loss,
            'sense_classification_loss': sense_classification_loss,
            'argument_classification_loss': argument_classification_loss,
            'metrics': metrics,
        }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        predicate_identification_loss = torch.stack([x['predicate_identification_loss'] for x in outputs]).mean()
        sense_classification_loss = torch.stack([x['sense_classification_loss'] for x in outputs]).mean()
        argument_classification_loss = torch.stack([x['argument_classification_loss'] for x in outputs]).mean()
        metrics = SimpleModel._compute_epoch_metrics(outputs)

        tensorboard_logs = {
            'val_loss': avg_loss,
            'val_loss_predicate_identification': predicate_identification_loss,
            'val_loss_sense_classification': sense_classification_loss,
            'val_loss_argument_classification': argument_classification_loss,
            'val_predicate_precision': metrics['predicates']['precision'],
            'val_predicate_recall': metrics['predicates']['recall'],
            'val_predicate_f1': metrics['predicates']['f1'],
            'val_sense_precision': metrics['senses']['precision'],
            'val_sense_recall': metrics['senses']['recall'],
            'val_sense_f1': metrics['senses']['f1'],
            'val_role_precision': metrics['roles']['precision'],
            'val_role_recall': metrics['roles']['recall'],
            'val_role_f1': metrics['roles']['f1'],
            'val_overall_precision': metrics['overall']['precision'],
            'val_overall_recall': metrics['overall']['recall'],
            'val_overall_f1': metrics['overall']['f1'],
        }

        return {
            'val_loss': avg_loss,
            'val_f1': metrics['overall']['f1'],
            'val_predicate_f1': metrics['predicates']['f1'],
            'val_sense_f1': metrics['senses']['f1'],
            'val_role_f1': metrics['roles']['f1'],
            'log': tensorboard_logs,
        }

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        metrics = SimpleModel._compute_epoch_metrics(outputs)

        return {
            'test_loss': avg_loss,
            'test_f1': metrics['overall']['f1'],
            'test_predicate_metrics': metrics['predicates'],
            'test_sense_metrics': metrics['senses'],
            'test_role_metrics': metrics['roles'],
            'test_overall_metrics': metrics['overall'],
        }

    def compute_step_metrics(self, scores, labels, word_sense_mask=None):
        predicates_g = labels['predicates']
        predicates_p = torch.argmax(scores['predicates'], dim=-1)
        predicate_tp = (predicates_p[torch.logical_and(predicates_g >= 0, predicates_p == 1)] == predicates_g[torch.logical_and(predicates_g >= 0, predicates_p == 1)]).sum()
        predicate_fp = (predicates_p[torch.logical_and(predicates_g >= 0, predicates_p == 1)] != predicates_g[torch.logical_and(predicates_g >= 0, predicates_p == 1)]).sum()
        predicate_fn = (predicates_p[predicates_g == 1] != predicates_g[predicates_g == 1]).sum()

        senses_g = labels['senses']
        if word_sense_mask is None:
            senses_p = torch.argmax(scores['senses'], dim=-1)
        else:
            senses_p = torch.argmax(scores['senses'] * word_sense_mask, dim=-1)
        sense_tp = (senses_p[torch.logical_and(senses_g >= 0, senses_p >= 1)] == senses_g[torch.logical_and(senses_g >= 0, senses_p >= 1)]).sum()
        sense_fp = (senses_p[torch.logical_and(senses_g >= 0, senses_p >= 1)] != senses_g[torch.logical_and(senses_g >= 0, senses_p >= 1)]).sum()
        sense_fn = (senses_p[senses_g >= 1] != senses_g[senses_g >= 1]).sum()

        roles_g = labels['roles']
        if self.hparams.argument_classifier == 'softmax':
            roles_p = torch.argmax(scores['roles'], dim=-1)
        role_tp = (roles_p[torch.logical_and(roles_g >= 0, roles_p >= 1)] == roles_g[torch.logical_and(roles_g >= 0, roles_p >= 1)]).sum()
        role_fp = (roles_p[torch.logical_and(roles_g >= 0, roles_p >= 1)] != roles_g[torch.logical_and(roles_g >= 0, roles_p >= 1)]).sum()
        role_fn = (roles_p[roles_g >= 1] != roles_g[roles_g >= 1]).sum()

        return {
            'predicate_tp': predicate_tp,
            'predicate_fp': predicate_fp,
            'predicate_fn': predicate_fn,
            'sense_tp': sense_tp,
            'sense_fp': sense_fp,
            'sense_fn': sense_fn,
            'role_tp': role_tp,
            'role_fp': role_fp,
            'role_fn': role_fn,
        }

    @staticmethod
    def _compute_classification_loss(scores, labels, num_classes, ignore_index=-1, weight=None, mask=None):
        if mask is not None:
            scores = scores + (mask + 1e-10).log()
        classification_loss = F.cross_entropy(
            scores.view(-1, num_classes),
            labels.view(-1),
            reduction='sum',
            weight=weight,
            ignore_index=ignore_index)

        return classification_loss

    @staticmethod
    def _compute_epoch_metrics(outputs):
        predicate_tp = torch.stack([o['metrics']['predicate_tp'] for o in outputs]).sum()
        predicate_fp = torch.stack([o['metrics']['predicate_fp'] for o in outputs]).sum()
        predicate_fn = torch.stack([o['metrics']['predicate_fn'] for o in outputs]).sum()

        predicate_precision = torch.true_divide(predicate_tp, (predicate_tp + predicate_fp)) if predicate_tp + predicate_fp > 0 else torch.as_tensor(0)
        predicate_recall = torch.true_divide(predicate_tp, (predicate_tp + predicate_fn)) if predicate_tp + predicate_fn > 0 else torch.as_tensor(0)
        predicate_f1 = 2 * torch.true_divide(predicate_precision * predicate_recall, predicate_precision + predicate_recall) if predicate_precision + predicate_recall > 0 else torch.as_tensor(0)

        sense_tp = torch.stack([o['metrics']['sense_tp'] for o in outputs]).sum()
        sense_fp = torch.stack([o['metrics']['sense_fp'] for o in outputs]).sum()
        sense_fn = torch.stack([o['metrics']['sense_fn'] for o in outputs]).sum()

        sense_precision = torch.true_divide(sense_tp, (sense_tp + sense_fp)) if sense_tp + sense_fp > 0 else torch.as_tensor(0)
        sense_recall = torch.true_divide(sense_tp, (sense_tp + sense_fn)) if sense_tp + sense_fn > 0 else torch.as_tensor(0)
        sense_f1 = 2 * torch.true_divide(sense_precision * sense_recall, sense_precision + sense_recall) if sense_precision + sense_recall > 0 else torch.as_tensor(0)

        role_tp = torch.stack([o['metrics']['role_tp'] for o in outputs]).sum()
        role_fp = torch.stack([o['metrics']['role_fp'] for o in outputs]).sum()
        role_fn = torch.stack([o['metrics']['role_fn'] for o in outputs]).sum()

        role_precision = torch.true_divide(role_tp, (role_tp + role_fp)) if role_tp + role_fp > 0 else torch.as_tensor(0)
        role_recall = torch.true_divide(role_tp, (role_tp + role_fn)) if role_tp + role_fn > 0 else torch.as_tensor(0)
        role_f1 = 2 * torch.true_divide(role_precision * role_recall, role_precision + role_recall) if role_precision + role_recall > 0 else torch.as_tensor(0)

        overall_tp = role_tp + sense_tp
        overall_fp = role_fp + sense_fp
        overall_fn = role_fn + sense_fn
        overall_precision = torch.true_divide(overall_tp, (overall_tp + overall_fp)) if overall_tp + overall_fp > 0 else torch.as_tensor(0)
        overall_recall = torch.true_divide(overall_tp, (overall_tp + overall_fn)) if overall_tp + overall_fn > 0 else torch.as_tensor(0)
        overall_f1 = 2 * torch.true_divide(overall_precision * overall_recall, overall_precision + overall_recall) if overall_precision + overall_recall > 0 else torch.as_tensor(0)

        return {
            'predicates': {
                '_tp': predicate_tp,
                '_fp': predicate_fp,
                '_fn': predicate_fn,
                'precision': predicate_precision,
                'recall': predicate_recall,
                'f1': predicate_f1,
            },
            'senses': {
                '_tp': sense_tp,
                '_fp': sense_fp,
                '_fn': sense_fn,
                'precision': sense_precision,
                'recall': sense_recall,
                'f1': sense_f1,
            },
            'roles': {
                '_tp': role_tp,
                '_fp': role_fp,
                '_fn': role_fn,
                'precision': role_precision,
                'recall': role_recall,
                'f1': role_f1,
            },
            'overall': {
                '_tp': overall_tp,
                '_fp': overall_fp,
                '_fn': overall_fn,
                'precision': overall_precision,
                'recall': overall_recall,
                'f1': overall_f1,
            },
        }

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--input_representation', type=str, default='bert_embeddings')
        parser.add_argument('--word_embedding_size', type=int, default=300)
        parser.add_argument('--word_projection_size', type=int, default=512)
        parser.add_argument('--word_dropout', type=float, default=0.5)
        parser.add_argument('--vocab_size', type=int, default=30_000)
        parser.add_argument('--min_word_frequency', type=int, default=3)
        parser.add_argument('--language_model', type=str, default='bert-base-multilingual-cased')
        parser.add_argument('--language_model_fine_tuning', default=False, action='store_true')

        parser.add_argument('--lowercase', dest='lowercase', action='store_true')
        parser.add_argument('--no_lowercase', dest='lowercase', action='store_false')
        parser.set_defaults(lowercase=False)
        parser.add_argument('--fasttext_path', type=str, default='../embeddings/fasttext_en.txt')

        parser.add_argument('--sequence_representation', type=str, default='connected_lstm')
        parser.add_argument('--lstm_hidden_size', type=int, default=512)
        parser.add_argument('--lstm_num_layers', type=int, default=3)
        parser.add_argument('--lstm_dropout', type=float, default=0.2)
        parser.add_argument('--lstm_bidirectional', dest='lstm_bidirectional', action='store_true')
        parser.add_argument('--lstm_unidirectional', dest='lstm_bidirectional', action='store_false')
        parser.set_defaults(lstm_bidirectional=True)

        parser.add_argument('--predicate_encoding_size', type=int, default=128) # 32
        parser.add_argument('--predicate_encoder_layers', type=int, default=1)
        parser.add_argument('--predicate_encoder_activation', type=str, default='swish')
        parser.add_argument('--predicate_dropout_rate', type=float, default=0.5)

        parser.add_argument('--sense_encoding_size', type=int, default=512) # 256
        parser.add_argument('--sense_encoder_layers', type=int, default=1)
        parser.add_argument('--sense_encoder_activation', type=str, default='swish')
        parser.add_argument('--sense_dropout_rate', type=float, default=0.2)

        parser.add_argument('--predicate_timestep_encoding_size', type=int, default=512)
        parser.add_argument('--predicate_timestep_encoder_layers', type=int, default=1)
        parser.add_argument('--predicate_timestep_encoder_activation', type=str, default='swish')
        parser.add_argument('--predicate_timestep_dropout_rate', type=float, default=0.2)

        parser.add_argument('--argument_timestep_encoding_size', type=int, default=512)
        parser.add_argument('--argument_timestep_encoder_layers', type=int, default=1)
        parser.add_argument('--argument_timestep_encoder_activation', type=str, default='swish')
        parser.add_argument('--argument_timestep_dropout_rate', type=float, default=0.2)

        parser.add_argument('--argument_encoding_size', type=int, default=512)
        parser.add_argument('--argument_encoder_layers', type=int, default=1)
        parser.add_argument('--argument_encoder_activation', type=str, default='swish')
        parser.add_argument('--argument_dropout_rate', type=float, default=0.0)

        parser.add_argument('--argument_sequence_representation', type=str, default='connected_lstm')
        parser.add_argument('--argument_lstm_hidden_size', type=int, default=512) # 256
        parser.add_argument('--argument_lstm_num_layers', type=int, default=1) # 2
        parser.add_argument('--argument_lstm_dropout', type=float, default=0.2)
        parser.add_argument('--argument_lstm_bidirectional', dest='argument_lstm_bidirectional', action='store_true')
        parser.add_argument('--argument_lstm_unidirectional', dest='argument_lstm_bidirectional', action='store_false')
        parser.set_defaults(argument_lstm_bidirectional=True)

        parser.add_argument('--argument_classifier', type=str, default='softmax')

        parser.add_argument('--warmup_epochs', type=float, default=1.0)
        parser.add_argument('--cooldown_epochs', type=int, default=15)
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        parser.add_argument('--min_learning_rate', type=float, default=5e-5)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--language_model_learning_rate', type=float, default=1e-3)
        parser.add_argument('--language_model_min_learning_rate', type=float, default=5e-5)
        parser.add_argument('--language_model_weight_decay', type=float, default=1e-4)
        return parser
